
#!/usr/bin/env python3

import pyttsx3
import time

#Make the computer talk to us
def say(objects_list, scene, dimensions):

    cleaned_items = [item.replace("_", " ") for item in objects_list]

    pyttsx3.speak((f"We are looking at the scene {scene} and I can recognize {len(objects_list)} objects. Let's start with"))

    for i in range(len(dimensions)):
        dim = dimensions[i]
        item = cleaned_items[i]
        pyttsx3.speak(f"the object number {int(i + 1)}. The {item}, has dimensions {round(dim[0], 2)} by {round(dim[1], 2)}.")

        time.sleep(1)

    pyttsx3.speak("Thank you for listening, hope I did not miss anything.")
