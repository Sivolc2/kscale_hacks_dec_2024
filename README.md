# PROJECT ZERO - Voice-Controlled Robot with Digital Twin

A hackathon project that combines voice control, computer vision, and digital twin capabilities to create an extensible robot control system.

## Overview

This project enables:

- Voice control of a robot using natural language commands
- Real-time computer vision monitoring and feedback
- Digital twin simulation for testing and creating new skills
- Extensible skill library that can be expanded through demonstration

## Key Components

### Robot Monitor (`robot_monitor.py`)
- Main control loop integrating voice, vision and robot control
- Processes voice commands using Whisper
- Captures and analyzes camera frames
- Executes skills based on voice/vision input

### Digital Twin (`digital_twin.py`) 
- MuJoCo-based simulation environment
- Supports bi-directional sync between real robot and simulation
- Record and playback capabilities for creating new skills
- Three modes:
  - `sim-to-robot`: Simulation controls robot
  - `robot-to-sim`: Robot movements reflected in sim
  - `playback`: Replay recorded movements

### Skill Library (`skill_library.py`)
- Extensible library of robot behaviors
- Skills can be created through digital twin recording
- Each skill has clear objectives and validation
- Platform-agnostic design

### Model Handler (`model_handler.py`)
- Uses Claude 3 Sonnet for:
  - Scene analysis
  - Task planning
  - Movement validation
- Provides natural language interface

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Configure settings in `config/config.yaml`:
```yaml
monitor:
    capture_interval: 5.0
    save_dir: "pictures/webcam"
    platform: "zeroth"
    voice_mode: true
    whisper_model: "small"
```
3. Run the robot monitor:
```bash
python robot_monitor.py
```

4. Create new skills using digital twin:
```bash
python digital_twin.py
```


## Creating New Skills

1. Launch digital twin in `robot-to-sim` mode
2. Press 'R' to start recording
3. Move robot through desired motion sequence
4. Press 'R' again to save recording
5. New skill will be saved to `episodes/` directory
6. Add skill definition to `skill_library.py`

## Voice Commands

The robot responds to commands starting with "zero" or "hey zero". Examples:

- "Zero, please stand up"
- "Hey zero, walk forward"
- "Zero, wave hello"

## Requirements

- Python 3.8+
- MuJoCo 2.3.3+
- OpenCV
- PyTorch
- Anthropic API key (for Claude)
- Webcam/Camera
- Compatible robot hardware

## Project Structure

```bash
.
├── config/
│ └── config.yaml
├── robot_monitor.py # Main control loop
├── digital_twin.py # Simulation environment
├── skill_library.py # Robot behaviors
├── model_handler.py # AI model integration
├── robot.py # Hardware interface
├── image_processor.py # Vision processing
├── voice_processor.py # Speech processing
└── episodes/ # Recorded skills
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add new skills or improvements
4. Submit a pull request

## License

MIT License

## Acknowledgments

- MuJoCo physics engine
- Claude 3 Sonnet
- OpenAI Whisper
- Anthropic API


