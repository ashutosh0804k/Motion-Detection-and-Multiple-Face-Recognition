# Motion Detection and Multiple Faces Identification using Webcam

A real-time computer vision system that combines motion detection with multiple face identification capabilities using machine learning techniques and webcam input.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![MySQL](https://img.shields.io/badge/MySQL-Database-orange.svg)](https://www.mysql.com/)

## 📋 Project Overview

This project implements an advanced security system that leverages computer vision and machine learning to detect motion and identify multiple faces simultaneously in real-time video streams. The system provides automated surveillance capabilities with minimal human intervention, making it ideal for security applications in homes, organizations, and public spaces.

## 👥 Team Members

- **Varad Kalkhaire** (B190724251)
- **Ashutosh Khatavkar** (B190724262)
- **Shubham Dhanne** (B190724224)
- **Shantanu Bhumkar** (B190724213)

**Institution**: K J College of Engineering and Management Research  
**University**: Savitribai Phule Pune University  
**Guide**: Prof. Rohini V. Agawane  
**Department**: Computer Engineering  
**Academic Year**: 2022-2023

## 🎯 Key Features

### Motion Detection
- **Real-time motion tracking** using Gaussian Blur method
- **Background subtraction** for identifying moving objects
- **Automated alerts** when motion is detected
- **Frame-by-frame analysis** for accurate detection
- **Timestamp logging** of detected movements

### Multiple Face Identification
- **Simultaneous detection** of multiple faces in a single frame
- **Face recognition** using stored facial encodings
- **Database-driven identification** with MySQL integration
- **Automatic face registration** for unknown individuals
- **Name update functionality** for registered faces
- **Haar Cascade Classifier** for efficient face detection

### Advanced Capabilities
- **Webcam/CCTV integration** for live video streaming
- **Grayscale conversion** for optimized processing
- **Feature extraction** using facial landmarks
- **Real-time video stream processing**
- **User-friendly GUI** for system management

## 🛠️ Technical Architecture

### Core Technologies

#### 1. **Gaussian Blur**
- Noise reduction in video frames
- Image smoothing for better detection
- Convolutional kernel-based filtering
- Supports background subtraction

#### 2. **Haar Cascade Classifier**
- Machine learning-based face detection
- Trained on positive and negative image samples
- Uses integral images for fast computation
- Haar-like features for pattern recognition
- Cascade function for efficient processing

#### 3. **Face Recognition Module**
- Deep learning-based facial embeddings
- 128-dimensional face encoding vectors
- Distance-based matching algorithm
- Support for multiple face database management
- Real-time recognition capabilities

## 📊 System Components

### Block Diagram Flow
```
Camera → Video Stream → Frame Extraction → 
Grayscale Conversion → Motion Detection (Gaussian Blur) → Alert System
                     ↓
              Face Detection (Haar Cascade) → 
              Feature Extraction → Database Comparison → 
              Face Recognition → Display Results
```

### Database Schema

**Unknown Table**
- face_enc (BLOB) - Facial encoding data
- img_name (VARCHAR) - Image filename

**Known Table**
- face_enc (BLOB) - Facial encoding data
- name (VARCHAR) - Person's name

## 📦 Installation

### Prerequisites
```bash
Python 3.x
MySQL Server
Webcam or IP Camera
```

### Required Libraries
```bash
# Install dependencies
pip install opencv-python
pip install face-recognition
pip install mysql-connector-python
pip install numpy
pip install Pillow
```

### Database Setup
```sql
-- Create database
CREATE DATABASE ImageDB;

-- Create tables
USE ImageDB;

CREATE TABLE unknown (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_enc BLOB NOT NULL,
    img_name VARCHAR(255) NOT NULL
);

CREATE TABLE known (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_enc BLOB NOT NULL,
    name VARCHAR(255) NOT NULL UNIQUE
);
```

### Project Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/motion-face-detection.git
cd motion-face-detection

# Configure database connection
# Edit config file with your MySQL credentials
# Update: host, username, password, database name

# Run the application
python main.py
```

## 🎮 Usage

### Starting the System

```python
from motion_detection import MotionDetectionSystem

# Initialize the system
system = MotionDetectionSystem()

# Start motion detection and face identification
system.start()
```

### GUI Features

1. **Motion Detection and Face Identification**
   - Main interface showing live video stream
   - Real-time motion detection with green bounding boxes
   - Face detection with yellow rectangles
   - Name display for recognized faces (red labels)

2. **Face Registration**
   - Register unknown faces detected by the system
   - Enter person's name for the detected face
   - Automatic validation for duplicate names
   - Success confirmation messages

3. **Name Update**
   - Search for registered persons
   - Update existing names in the database
   - Face preview for confirmation
   - Duplicate name prevention

## 📈 Performance

### Test Results

| Component | Accuracy | Speed | Status |
|-----------|----------|-------|--------|
| Motion Detection | 95%+ | Real-time | ✓ Pass |
| Face Detection | 90%+ | <100ms/frame | ✓ Pass |
| Face Recognition | 88%+ | <200ms/face | ✓ Pass |
| Database Operations | 100% | <50ms | ✓ Pass |

### System Requirements

- **RAM**: 8GB (minimum)
- **Storage**: 500GB HDD/SSD
- **Processor**: Intel i5 or AMD Ryzen 5 (or above)
- **Camera**: System Webcam or IP Camera (720p minimum)
- **OS**: Windows 10+ or Ubuntu 18.04+

## 🔬 Algorithm Workflow

### Motion Detection Algorithm
```
1. Capture video stream from webcam
2. Extract frames continuously
3. Convert frames to grayscale
4. Apply Gaussian Blur (noise reduction)
5. Calculate absolute difference between consecutive frames
6. Apply threshold to detect significant changes
7. Identify motion regions of interest
8. Alert system if motion detected
9. Log timestamp and movement duration
```

### Face Identification Algorithm
```
1. Load Haar Cascade classifier
2. Detect faces in current frame
3. Extract facial region of interest (ROI)
4. Align face to standardized pose
5. Generate 128-dimensional face encoding
6. Fetch known face encodings from database
7. Compare current encoding with database
8. Calculate Euclidean distance
9. If distance < threshold: Identify person
10. If distance > threshold: Mark as "Unknown" and store
11. Display results on video frame
```

## 🌟 Applications

### Security and Surveillance
- Building access control
- Perimeter monitoring
- Intrusion detection
- Visitor management systems

### Home Automation
- Smart doorbell integration
- Automated lighting based on occupancy
- Family member recognition
- Elderly care monitoring

### Healthcare
- Patient monitoring in hospitals
- Fall detection systems
- Mental health facility security
- Visitor verification

### Retail and Marketing
- Customer analytics
- Demographic tracking
- VIP customer identification
- Theft prevention

### Transportation
- Airport security
- Boarding process automation
- Lost passenger identification
- Baggage tracking

### Education
- Automated attendance systems
- Campus security
- Exam invigilation
- Library access control

## ✅ Advantages

1. **Enhanced Security**: Automated 24/7 monitoring with immediate alerts
2. **Real-Time Processing**: Instant detection and identification
3. **Cost-Effective**: Uses standard webcams instead of specialized hardware
4. **Scalable**: Easily expandable to multiple camera systems
5. **Non-Intrusive**: No physical contact required
6. **Automated**: Minimal human intervention needed
7. **Database-Driven**: Efficient storage and retrieval of face data
8. **Flexible**: Adaptable to various environments and lighting conditions

## ⚠️ Limitations

1. **Limited Field of View**: Webcam coverage area constraints
2. **Image Quality Dependency**: Performance varies with camera resolution
3. **Lighting Sensitivity**: Accuracy affected by poor lighting conditions
4. **Processing Power**: Requires adequate computational resources
5. **Occlusion Challenges**: Partial face coverage affects recognition
6. **Privacy Concerns**: Requires proper data protection measures
7. **False Positives**: Environmental factors can trigger false alerts

## 🔮 Future Enhancements

1. **Deep Learning Integration**
   - CNN-based face recognition
   - Improved accuracy with larger datasets
   - Emotion detection capabilities

2. **Advanced Features**
   - Age and gender estimation
   - Mask detection (COVID-19 compliance)
   - Behavioral analysis
   - Suspicious activity detection

3. **System Improvements**
   - Multi-camera synchronization
   - Cloud storage integration
   - Mobile app for remote monitoring
   - Edge computing for reduced latency

4. **AI Enhancements**
   - Predictive analytics
   - Anomaly detection
   - Crowd density estimation
   - Social distancing monitoring

5. **Integration Capabilities**
   - IoT device integration
   - Smart home system compatibility
   - Third-party security system integration
   - API for external applications

## 📚 References

1. Khan, M., Chakraborty, S., Astya, R., & Khepra, S. (2019). "Face Recognition using OpenCV". IEEE.

2. Huang, S., & Luo, H. (2020). "Attendance System Based on Dynamic Face Detection". IEEE.

3. Zhaoyang, C., Haolin, G., & Kun, W. (2020). "A Motion-Based Object Detection Method". IEEE.

4. Sharma, V. K. (2019). "Designing of Face Recognition System". IEEE.

5. Singh, N., Kumar, P., Akhoury, P., Kumar, R., & Ramasubramanian, M. (2019). "Motion Detection Application Using Web Camera". IJMER.

6. Tathe, S. V., Narote, A. S., & Narote, S. P. (2016). "Human Face Detection and Recognition in Videos". ICACCI.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is part of academic coursework at K J College of Engineering and Management Research.

## 🙏 Acknowledgments

- Prof. Rohini V. Agawane for project guidance
- Prof. Aparna S. Hambarde, Project Coordinator
- Dr. Nikita Kulkarni, Head of Department
- Dr. Suhas Khot, Principal
- Savitribai Phule Pune University
- K J College of Engineering and Management Research

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Varad Kalkhaire**: [email]
- **Ashutosh Khatavkar**: [email]
- **Shubham Dhanne**: [email]
- **Shantanu Bhumkar**: [email]

---

**Note**: This system is designed for legitimate security and monitoring purposes. Users must ensure compliance with local privacy laws and regulations when deploying this technology.
