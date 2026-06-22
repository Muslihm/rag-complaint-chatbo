import pyautogui
import time
import os

# Create directory
os.makedirs('docs/screenshots', exist_ok=True)

print("Taking screenshots in 5 seconds...")
time.sleep(5)

# Take screenshots of different app states
print("Capturing screenshots...")

# 1. Home page
pyautogui.screenshot('docs/screenshots/01_home.png')
print("1. Home page captured")

# Wait for user interaction
time.sleep(2)

# 2. Asking a question
pyautogui.screenshot('docs/screenshots/02_asking_question.png')
print("2. Question input captured")

time.sleep(2)

# 3. AI Response
pyautogui.screenshot('docs/screenshots/03_ai_response.png')
print("3. AI response captured")

time.sleep(2)

# 4. Sources
pyautogui.screenshot('docs/screenshots/04_sources.png')
print("4. Sources display captured")

time.sleep(2)

# 5. Product Filter
pyautogui.screenshot('docs/screenshots/05_product_filter.png')
print("5. Product filter captured")

time.sleep(2)

# 6. Multiple questions
pyautogui.screenshot('docs/screenshots/06_multiple_questions.png')
print("6. Multiple questions captured")

print("\n✅ Screenshots saved to: docs/screenshots/")

