# PROJECT ZERO - Building a More Capable Robot Through Skill Sharing

Over an intense 36-hour hackathon, our team set out to revolutionize how robots learn and share skills. We envisioned a future where robots could not only learn from human demonstrations but also from each other, creating a growing library of capabilities that any compatible robot could tap into.

<img src="pictures/complete_diagram.drawio.png" width="800" alt="System Architecture Diagram">


## Our Journey

<div align="center">
  <img src="pictures/robot_demo.jpg" width="400" alt="Robot Demo">
  <p><i>Project Zero in action during the hackathon</i></p>
</div>

What started as a wild idea turned into something incredible. We combined voice control, computer vision, and a digital twin system to create a robot that's not just programmable, but truly interactive. Imagine speaking to your robot as naturally as you would to a friend, and watching it understand and execute complex tasks. That's what we achieved with Project Zero.

The heart of our project lies in its extensible skills framework. We wanted to break away from the traditional model where each robot operates in isolation. Instead, we built a platform where skills can be created through various methods - from direct demonstration to reinforcement learning - and shared across different robots.

## The Hardware Stack

<div style="display: flex; justify-content: space-between;">
  <img src="pictures/modular_hand.png" width="300" alt="Modular Hand CAD">
  <img src="pictures/power_supply.jpg" width="300" alt="Custom Power Supply">
</div>

One of our proudest achievements is the modular tool system we designed. After countless hours in OnShape and several 3D printing iterations, we created an elegant quick-attach mechanism that allows our robot to use different tools. Our first success was a custom-designed beer can holder (because why not?), but we've got plans for more practical attachments like hooks, whisks, and forks. The possibilities are endless! The following is a CAD file of our modular hand: https://cad.onshape.com/documents/c793fce9991582313080b3a9/w/49260d9832ed2563113c8700/e/cf6d5ce4b4362395c19bbd5c?configuration=default&renderMode=0&uiState=675f5fbabe4315572f93ba0c

We also tackled the power problem head-on by building our own power supply from scratch. It wasn't just about providing juice to the motors - we needed something that could handle the demands of real-world operation while being reliable enough for continuous use.

## The Software Stack

<img src="pictures/software_stack.png" width="700" alt="Software Architecture">

Our software architecture is where things get really exciting. We integrated:
- A MuJoCo-based digital twin for testing and skill creation
- Voice control using Whisper for natural language commands
- Claude 3 Sonnet for high-level task understanding and planning
- Custom PPO implementations for learning from demonstrations

The real breakthrough came when we got all these systems working together seamlessly. You can now teach the robot new skills in multiple ways:
- Demonstrate the movement directly and record it
- Create precise movements in simulation
- Let the robot learn through reinforcement learning
- Or even combine these methods for more complex behaviors

## Looking Forward

While we're incredibly proud of what we accomplished in 36 hours, we see this as just the beginning. Our modular design and extensible skill library create endless possibilities for future development. We're excited to see how the community might build upon this foundation to create even more capable robots. We would love to continue to extend the possibility of the the Zeroth bot for future developers and roboticians

## Technical Deep Dive

### The Magic of Our Digital Twin
The digital twin environment is one of our favorite achievements. Built on MuJoCo, it provides a perfect sandbox for testing new ideas. Want to try a risky movement sequence? Test it in simulation first! The bi-directional sync means you can:
- Watch your real robot's movements mirrored in real-time
- Test new skills safely before deploying them
- Record and replay complex sequences
- Train learning algorithms without risking hardware

### Voice Control & Natural Language Understanding
"Hey Zero, could you grab that can?" - our robot is able to respond to speech commands. We integrated Whisper for speech recognition and Claude 3 Sonnet for understanding intent. The system can:
- Parse complex natural language commands
- Understand context from visual input
- Break down complex tasks into simple movements
- Provide verbal feedback on task progress

# Acknowledgements
This project taught us so much more than we expected. We dove deep into the Zeroth bot SDK, wrestled with IMU calibration issues (thank you, Kelsey!), and learned the intricacies of battery management (couldn't have done it without JX's guidance). Denys was invaluable in helping us fine-tune our calibration processes.

The K-Scale and Zeroth bot teams provided incredible support throughout the hackathon. Their simulation environment and robust hardware platform gave us the foundation we needed to build something truly special.

