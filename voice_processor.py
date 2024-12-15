#! python3.7

import argparse
import os
import speech_recognition as sr
import whisper
import torch
import numpy as np
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from abc import ABC, abstractmethod
import time
from typing import Optional, List, Dict, Any
import queue
import threading
from dataclasses import dataclass, field
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from enum import Enum

class CommandState(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Command:
    """Represents a voice command with state tracking"""
    text: str
    timestamp: datetime
    state: CommandState = CommandState.PENDING
    context: List[str] = field(default_factory=list)
    response: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VoiceContext:
    """Stores context about the voice interaction"""
    timestamp: datetime
    text: str
    is_command: bool = False
    processed: bool = False
    command: Optional[Command] = None

class VoiceProcessor(ABC):
    """Abstract base class for voice processing"""
    
    def __init__(self, buffer_timeout: float = 30.0, command_timeout: float = 60.0):
        self.context_buffer: List[VoiceContext] = []
        self.buffer_timeout = buffer_timeout
        self.command_timeout = command_timeout
        self.listening = False
        
        # Separate queues for different command states
        self.pending_commands = queue.Queue()
        self.active_commands: Dict[datetime, Command] = {}
        self.completed_commands: List[Command] = []
        
        self.context_queue = queue.Queue()
        self._listen_thread = None
        self._stop_event = threading.Event()

    def start_listening(self):
        """Start background listening thread"""
        if not self.listening:
            self._stop_event.clear()
            self._listen_thread = threading.Thread(target=self._listen_loop)
            self._listen_thread.daemon = True
            self._listen_thread.start()
            self.listening = True
            print("Voice processor started listening")

    def stop_listening(self):
        """Stop background listening"""
        if self.listening:
            self._stop_event.set()
            if self._listen_thread:
                self._listen_thread.join(timeout=2)
            self.listening = False
            print("Voice processor stopped listening")

    def _listen_loop(self):
        """Background listening loop"""
        while not self._stop_event.is_set():
            try:
                text = self._transcribe_audio()
                if text:
                    timestamp = datetime.now()
                    
                    # Check if it's a command
                    is_command = any(text.lower().startswith(word) for word in 
                                   ["zero", "hey zero", "hey, zero"])
                    
                    # Get recent context
                    recent_context = self.get_recent_context(seconds=10)
                    
                    if is_command:
                        # Create command object
                        command = Command(
                            text=text,
                            timestamp=timestamp,
                            context=recent_context
                        )
                        
                        # Create context with command reference
                        context = VoiceContext(
                            timestamp=timestamp,
                            text=text,
                            is_command=True,
                            command=command
                        )
                        
                        # Add to queues
                        self.pending_commands.put(command)
                        self.context_buffer.append(context)
                        print(f"Command queued: {text}")
                        
                    else:
                        # Regular context
                        context = VoiceContext(
                            timestamp=timestamp,
                            text=text,
                            is_command=False
                        )
                        self.context_buffer.append(context)
                        self.context_queue.put(context)
                        print(f"Context: {text}")
                    
                    # Clean old context
                    self._cleanup_buffer()
                    
            except Exception as e:
                print(f"Error in listen loop: {e}")
            sleep(0.1)

    def _cleanup_buffer(self):
        """Remove old context entries"""
        now = datetime.now()
        self.context_buffer = [
            ctx for ctx in self.context_buffer
            if (now - ctx.timestamp).total_seconds() <= self.buffer_timeout
        ]

    def get_next_command(self) -> Optional[Command]:
        """
        Get next pending command if no command is currently active
        Returns:
            Optional[Command]: Next command to process, or None if no commands ready
        """
        # First check if any active commands have timed out
        self._check_command_timeouts()
        
        # If there are no active commands, get the next pending command
        if not self.active_commands:
            try:
                command = self.pending_commands.get_nowait()
                self.active_commands[command.timestamp] = command
                command.state = CommandState.IN_PROGRESS
                return command
            except queue.Empty:
                return None
        
        return None

    def update_command_state(self, command: Command, 
                           state: CommandState, 
                           response: Optional[str] = None):
        """Update command state and handle completion"""
        command.state = state
        if response:
            command.response = response
            
        if state in [CommandState.COMPLETED, CommandState.FAILED, CommandState.CANCELLED]:
            # Remove from active commands and add to completed
            if command.timestamp in self.active_commands:
                del self.active_commands[command.timestamp]
            self.completed_commands.append(command)
            
            # Print status
            status = "✓" if state == CommandState.COMPLETED else "✗"
            print(f"{status} Command {state.value}: {command.text}")
            if response:
                print(f"Response: {response}")

    def _check_command_timeouts(self):
        """Check for and handle timed out commands"""
        now = datetime.now()
        timed_out = []
        
        for timestamp, command in self.active_commands.items():
            if (now - timestamp).total_seconds() > self.command_timeout:
                timed_out.append(timestamp)
                self.update_command_state(
                    command,
                    CommandState.FAILED,
                    "Command timed out"
                )
        
        # Remove timed out commands
        for timestamp in timed_out:
            del self.active_commands[timestamp]

    def get_command_status(self) -> Dict[str, int]:
        """Get count of commands in each state"""
        status = {state.value: 0 for state in CommandState}
        
        # Count pending commands
        status['pending'] = self.pending_commands.qsize()
        
        # Count active commands
        for cmd in self.active_commands.values():
            status[cmd.state.value] += 1
            
        # Count completed commands
        for cmd in self.completed_commands:
            status[cmd.state.value] += 1
            
        return status

    def get_recent_context(self, seconds: float = 10.0) -> List[str]:
        """Get recent conversation context"""
        now = datetime.now()
        return [
            ctx.text for ctx in self.context_buffer
            if (now - ctx.timestamp).total_seconds() <= seconds
        ]

    @abstractmethod
    def _transcribe_audio(self) -> Optional[str]:
        """Implement actual audio transcription"""
        pass

class WhisperVoiceProcessor(VoiceProcessor):
    """Voice processor using Whisper for transcription"""
    
    def __init__(self, 
                 model_name: str = "medium", 
                 energy_threshold: int = 1000,
                 record_timeout: float = 2.0,
                 phrase_timeout: float = 3.0):
        super().__init__()
        self.model_name = model_name if model_name != "large" else "medium.en"
        self.audio_model = whisper.load_model(self.model_name)
        
        # Initialize speech recognizer
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False
        
        self.record_timeout = record_timeout
        self.phrase_timeout = phrase_timeout
        self.data_queue = Queue()
        self.source = None

    def initialize_microphone(self, device_name: Optional[str] = None):
        """Initialize microphone source"""
        if 'linux' in platform and device_name:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if device_name in name:
                    self.source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
        else:
            self.source = sr.Microphone(sample_rate=16000)
        
        if self.source:
            with self.source:
                self.recorder.adjust_for_ambient_noise(self.source)

    def _transcribe_audio(self) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if not self.data_queue.empty():
            # Convert audio data to numpy array
            audio_data = b''.join(self.data_queue.queue)
            self.data_queue.queue.clear()
            
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
            return result['text'].strip()
        return None

    def _record_callback(self, _, audio: sr.AudioData) -> None:
        """Callback for audio recording"""
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def start_listening(self):
        """Start listening with proper microphone initialization"""
        if not self.source:
            self.initialize_microphone()
        
        if self.source:
            self.recorder.listen_in_background(
                self.source, 
                self._record_callback,
                phrase_time_limit=self.record_timeout
            )
            super().start_listening()
        else:
            raise RuntimeError("No microphone source available")

class DummyVoiceProcessor(VoiceProcessor):
    """Dummy voice processor for testing"""
    
    def __init__(self):
        super().__init__()
        self.dummy_commands = [
            "robot wave to me",
            "that was good",
            "robot walk forward please",
            "be careful",
            "robot stop moving",
            "thanks",
            None  # Simulate no voice input
        ]
        self.command_index = 0
    
    def _transcribe_audio(self) -> Optional[str]:
        """Simulate voice transcription"""
        sleep(0.5)  # Simulate processing time
        text = self.dummy_commands[self.command_index]
        self.command_index = (self.command_index + 1) % len(self.dummy_commands)
        return text

def main():
    """Test voice processing functionality"""
    parser = argparse.ArgumentParser(description='Voice Processing Test')
    parser.add_argument('--mode', 
                       choices=['dummy', 'whisper'], 
                       default='whisper',
                       help='Voice processor mode')
    parser.add_argument('--model', 
                       default="small",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model to use")
    parser.add_argument('--device', 
                       help="Microphone device name (Linux only)")
    args = parser.parse_args()

    # Initialize voice processor
    if args.mode == 'whisper':
        processor = WhisperVoiceProcessor(model_name=args.model)
        if args.device:
            processor.initialize_microphone(args.device)
    else:
        processor = DummyVoiceProcessor()

    try:
        print(f"Starting {args.mode} voice processor...")
        processor.start_listening()
        
        # Test loop
        while True:
            # Get next command if available
            command = processor.get_next_command()
            if command:
                print(f"\nProcessing command: {command.text}")
                print("Recent context:")
                for ctx in command.context:
                    print(f"  - {ctx}")
                
                # Simulate command processing
                sleep(2)
                
                # Update command state (randomly succeed or fail for testing)
                import random
                if random.random() > 0.3:  # 70% success rate
                    processor.update_command_state(
                        command,
                        CommandState.COMPLETED,
                        "Command executed successfully"
                    )
                else:
                    processor.update_command_state(
                        command,
                        CommandState.FAILED,
                        "Failed to execute command"
                    )
                
                # Show command status
                status = processor.get_command_status()
                print("\nCommand Queue Status:")
                for state, count in status.items():
                    print(f"  {state}: {count}")
            
            sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping voice processor...")
    finally:
        processor.stop_listening()

if __name__ == "__main__":
    main()
