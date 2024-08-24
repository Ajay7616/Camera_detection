# Camera Detection Project

This project involves object detection, counting of individuals in video feeds, and tracking historical data on detected individuals. It integrates with camera sources and uses several libraries and tools for implementation.

## Key Features

- **Object Detection**: Utilizes YOLO (You Only Look Once) for real-time object detection.
- **Counting Individuals**: Counts the number of individuals detected in video feeds.
- **Historical Data Tracking**: Stores data on detected individuals, including their last visit time.
- **Integration with Camera Sources**: Connects to camera sources for real-time detection.

## Libraries and Tools Used

- **OpenCV**: For video processing and object detection.
- **YOLO**: A real-time object detection system.
- **NumPy**: For numerical operations.
- **SQLite**: For database management.
- **COCO**: Common Objects in Context dataset used for object detection.

## Project Structure

- **detection.py**: Main script for object detection and counting individuals in video feeds.
- **people_count.db**: SQLite database file to store historical data on detected individuals.
- **yolov3.cfg**: YOLO configuration file for object detection.
- **coco.names**: File containing the list of classes used in YOLO for object detection.

## Setup and Running

1. **Clone the Repository**
   - Clone this repository to your local machine.

2. **Install Dependencies**
   - Ensure you have Python installed.
   - Install the required Python libraries using pip:
     ```bash
     pip install opencv-python-headless numpy
     ```

3. **Configure YOLO**
   - Download the YOLOv3 weights from [YOLO's official site](https://pjreddie.com/darknet/yolo/) and place them in the project directory.

4. **Run the Detection Script**
   - Open your terminal or command prompt.
   - Navigate to the project directory where `detection.py` is located:
     ```bash
     cd /path/to/your/project
     ```
   - Run the detection script:
     ```bash
     python detection.py
     ```

5. **Access the Database**
   - The `people_count.db` file is used to store historical data. You can query this database using any SQLite database browser or management tool to view the stored data.

## Configuration

- **YOLO Configuration Files**: Ensure `yolov3.cfg` and `coco.names` are correctly placed in the project directory.
- **Database Configuration**: The SQLite database is automatically managed by the script. You may adjust the database path in `detection.py` if necessary.

## Conclusion

This project provides a comprehensive solution for object detection and individual counting using video feeds. By following the setup instructions, you can run the detection script and analyze the results. The integration with camera sources and historical data tracking makes it a versatile tool for various applications.

Feel free to explore and modify the project as needed for your specific use cases.
