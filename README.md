# New-Hacks-SNK

## Inspiration/ Problem to Solve
In disaster scenarios like the 2010 Haiti earthquake, where 300,000 people were injured, access to professional medical care is often limited. 
In disaster zones like these where traditional medical resources are scarce or delayed, MedScope aims to empower individuals to provide immediate aid in emergency situations by leveraging readily available objects in their surroundings.
MedScope bridges the gap between injury onsets and the arrival of professional medical teams by informing the users on how to improvise medical solutions using the materials available around them (ex. To use as stretchers, fabrics to use as bandages). 

## What it does
The application takes in an image of the surrounding area, and then scans it with OpenCV. The image is run through an object detection model that detects any objects within the scene and identifies them. After this we use Gemini’s API to provide the user with ideas on how to use the objects around them as medical devices in the case of injury of a person nearby.

## How we built it
Our project development involved three main stages: Input, Processing, and Output, each contributing essential functions to achieve real-time object detection and intelligent suggestions.

Input Stage:
The project starts by capturing an image, where it is processed. We applied image enhancement techniques to ensure clarity and accuracy in object detection. Each enhanced frame was then compressed into JPEG format to optimize storage without compromising essential details, creating a ready-to-process dataset.

Processing Stage:
In this stage, images are loaded and preprocessed to match the input requirements of the ResNet50 model, which we used for feature extraction and initial analysis. The ResNet50 model provided inference capabilities, allowing us to obtain confidence scores for detected objects. We filtered objects based on confidence thresholds to maintain accuracy, ensuring that only the most relevant objects were passed along to the next stage.

Output Stage:
The results of the object detection were then displayed in real-time, providing immediate feedback on identified objects. Additionally, we integrated the Gemini API to access external data and enhance functionality. Using this data, we generated intelligent suggestions for medical tools based on detected objects, making the system valuable in a medical setting.

Error Handling:
Throughout each stage, robust error handling mechanisms were integrated to manage potential issues, from video feed interruptions to API response delays, ensuring smooth and reliable performance.



## Challenges we ran into
One of the major hiccups we faced was downloading the required libraries for implementing the Gemini API because of the way Python libraries are downloaded on Macs. Another challenge within Image processing was due to real-world image variability, requiring continuous model tuning and optimization to improve accuracy. Creating a method to handle errors was essential to understand which parts of the code the software crashed.


## Accomplishments that we're proud of
Utilizing Gemini API: Successfully learned to interact with the Gemini API for various tasks.
Proficient in Image Processing: Gained expertise in using image processing libraries to manipulate and analyze images.
Effective Time Management: Completed the project in the given timeframe, with the additional optimizations and bug fixes.


## What we learned
One of the major things we learned was how to implement APIs for LLM models such as Gemini. Within this, we learned how to optimize and engineer prompts to be more efficient for what we want the model to accomplish. This was one of the most fundamental elements of our project and it was our first time ever using LLM APIs.

## What's next for MedScope
"To take MedScope to the next level, our priority is building a user-friendly and intuitive interface that makes navigation seamless, even under stress. This means designing a UI that’s visually simple, with easy-to-follow instructions and quick access to key features. On the technical side, we’re committed to training our model for higher accuracy in object detection, ensuring it can quickly and reliably identify a broad range of everyday items. We also plan to add functionality for processing real-time video streams, enabling users to scan surroundings continuously without having to pause for each item. Additionally, a feature allowing the app to process multiple images simultaneously will be essential for covering more ground and providing faster results in urgent situations. With these upgrades, MedScope will become a more powerful and efficient tool for real-time medical support."
