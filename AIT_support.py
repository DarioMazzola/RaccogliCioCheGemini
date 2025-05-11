import os
import time
import pyautogui
import re
import importlib.util
import platform
import threading
import traceback
from flask import Flask, render_template, request, jsonify
from PIL import Image
import google.generativeai as genai

# Get the current operating system
current_os = platform.system()  # Returns 'Linux', 'Windows', 'Darwin' (for macOS)
os_details = platform.platform()  # More detailed OS information

# Define log file
LOG_FILE = "llm_outputs.log"

# Configure the Flask app
app = Flask(__name__, static_folder='static')

# Configure Gemini API
genai.configure(api_key="API_KEY")
model = genai.GenerativeModel('gemini-2.0-flash')

# Create static folder for screenshots if it doesn't exist
os.makedirs('static/screenshots', exist_ok=True)

# Dictionary to track running automations
automation_tasks = {}

def clean_python_code(code_str):
    # Extract only the content between ```python and ```
    if "```python" in code_str and "```" in code_str[code_str.find("```python")+8:]:
        start_idx = code_str.find("```python") + len("```python")
        end_idx = code_str.find("```", start_idx)
        code_str = code_str[start_idx:end_idx]
    
    # Strip any leading/trailing whitespace
    code_str = code_str.strip()
    
    return code_str

def log_llm_output(model_name, output):
    with open(LOG_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n===== {model_name} =====\n")
        log_file.write(f"Timestamp: {time.ctime()}\n")
        log_file.write(f"Output:\n{output.strip()}\n")
        log_file.write("=" * 40 + "\n\n")

def ask_agent(markdown_guide):
    # Take a screenshot for context
    screenshot = pyautogui.screenshot()
    screenshot_path = "static/screenshots/temp_screenshot.png"
    screenshot.save(screenshot_path)
    image_file = Image.open(screenshot_path)
    
    prompt = f"""
    ## ROLE ##
    You are a Python automation expert specialized in using `pyautogui` to convert natural language task lists into fully executable automation scripts. Your purpose is to understand and translate the user's instructions into raw Python code that automates the steps exactly as described.
    The users are non-programmers or technical staff who describe their tasks in plain language. You must convert these into precise and runnable Python scripts.

    ## VISUAL CONTEXT ##
    You are also provided with a screenshot showing the **current screen state** before each step begins. You must interpret this image to determine where the UI currently is, so you can make only the necessary actions to reach the desired outcome efficiently.

    For example:
    - If the goal is to turn off Bluetooth and the screenshot shows the system settings already open, you should only navigate within that window.
    - If the user wants to search something on Google and the screenshot shows a browser open, you must use that context to identify the precise position where to click to begin the search.

    You must use the visual information to determine what is already open, what UI elements are visible, and where to move the mouse or click.

    ## REASONING ##
    Use the following format:
    Image: verify the UI state before the action
    Thought: analyze what you have to do given the screenshot provided
    Final Answer: your python code that executes the steps to reach the desired outcome. You must return **only** raw Python code.

    ## CONSTRAINTS ##
    - Wrap the code in markdown formatting (```python).
    - **Do not** include any text, comments, explanations, or metadata.
    - Each task step must be implemented as a separate Python function using the format `step_1()`, `step_2()`, etc.
    - Use `time.sleep(0.5)` for short pauses; avoid long or unnecessary delays.
    - Follow the task list exactly and convert each instruction into corresponding `pyautogui` actions.
    - The code must be directly executable in the user's operating system environment.
    - If a function is impossible to generate or ambiguous, you may skip it entirely—do not add placeholders or explanations.

    ## INPUT CONTEXT ##
    List instructions:
    {markdown_guide}

    Operating System: {current_os}  
    OS Details: {os_details}

    ###
    Begin!
    """
    
    response = model.generate_content([prompt, image_file])
    print("LLM Response:", response.text)
    log_llm_output("gemini-2.5-pro", response.text)
    return response.text

def ask_agent_for_verification_with_comparison(before_screenshot_path, after_screenshot_path, step_description):
    before_image_data = Image.open(before_screenshot_path)
    after_image_data = Image.open(after_screenshot_path)

    prompt = f"""
    ## ROLE ##
    You are an expert in evaluating user interface (UI) automation steps. You specialize in analyzing visual differences in UI states before and after an action is taken, with deep knowledge of operating systems, UI frameworks, and common user interface behaviors and patterns.

    Your task is to determine whether a described UI step was successfully executed, using only the visual evidence provided in the screenshots.

    ## REASONING ##
    Use the following format:
    Input: the step the user executed  
    Thought: analyze what the step is intended to do and what visual indicators would confirm its success  
    Before Image: verify whether the 'Before' image shows the expected UI state before the action  
    After Image: verify whether the 'After' image shows the expected UI state after the action  
    Final Answer: your final evaluation — must be **true** if the step was completed successfully, **false** otherwise

    ## IMAGES ##
    You will receive two screenshots:
    - The first image is the UI **before** the action
    - The second image is the UI **after** the action

    - Example -
    If the user says: "I clicked the 'Submit' button to send the form"  
    You must interpret the intended outcome (e.g., form disappears, confirmation message appears), compare the before and after screenshots, and decide if the visual result confirms the step's success.

    ## CONSTRAINTS ##
    You must:
    - Always use the labels: Thought, Before Image, After Image, Final Answer
    - Base your judgment strictly on what is visually evident in the screenshots
    - Return only 'true' or 'false' as the Final Answer
    - Take into account the operating system and UI context provided
    - Never reference tools, screenshots, or this prompt in the Final Answer
    - Always use the same language as the user input

    ###
    Begin!

    Input: I just executed this step in a UI automation task: "{step_description}"  
    Operating System: {current_os}  
    OS Details: {os_details}
    """
    
    response = model.generate_content([prompt, before_image_data, after_image_data])
    print("LLM Response:", response.text)
    log_llm_output("gemini-2.5-pro", response.text)
    result = response.text.strip().lower()
    return "true" in result

def ask_agent_to_fix_step_with_comparison(before_screenshot_path, after_screenshot_path, step_description, step_index):
    before_image_data = Image.open(before_screenshot_path)
    after_image_data = Image.open(after_screenshot_path)

    prompt = f"""
    ## ROLE ##
    You are an expert in evaluating user interface (UI) automation steps. You specialize in analyzing visual differences in UI states before and after an action is taken, with deep knowledge of operating systems, UI frameworks, and common user interface behaviors and patterns.

    Your task is to determine whether a described UI step was successfully executed, using only the visual evidence provided in screenshots.

    ## REASONING ##
    Use the following format:
    Input: the step the user executed  
    Thought: analyze what the step is intended to do and what visual indicators would confirm its success  
    Before Image: verify whether the first image ("before") shows the expected UI state before the action  
    After Image: verify whether the second image ("after") shows the expected UI state after the action  
    Final Answer: your final evaluation — must be **true** if the step was completed successfully, **false** otherwise

    ## IMAGES ##
    You will receive two screenshots:
    - The first image is the UI **before** the action
    - The second image is the UI **after** the action

    ## CONSTRAINTS ##
    You must:
    - Always use the labels: Thought, Before Image, After Image, Action, Final Answer
    - Base your judgment strictly on what is visually evident in the screenshots
    - Take into account the operating system and UI context provided
    - Never reference tools, screenshots, or this prompt in the Final Answer
    - Always use the same language as the user input
    - Output **only** executable Python code for a function named `step_{step_index+1}()`.
    - Not include any explanations, markdown, comments, or extra formatting—only the Python code.
    - Ensure the function reflects the intended step precisely, using only pyautogui and standard Python.
    - Adapt your function based on the given operating system details.
    - Rely solely on the visual and descriptive context to reconstruct the correct behavior.

    ###
    Begin!

    Input: I failed to execute this step in a UI automation task: "{step_description}"  
    Operating System: {current_os}  
    OS Details: {os_details}
    """
    
    response = model.generate_content([prompt, before_image_data, after_image_data])
    print("LLM Response:", response.text)
    log_llm_output("gemini-2.5-pro", response.text)
    # Clean up the response to remove any markdown formatting
    cleaned_code = clean_python_code(response.text.strip())
    return cleaned_code

def execute_step(task_id, step_fn, step_index, steps):
    add_log(task_id, f"Executing Step {step_index + 1}: {step_fn.__name__}")
    
    # Take screenshot BEFORE executing the step
    before_screenshot_path = f"static/screenshots/{task_id}_step_{step_index + 1}_before.png"
    before_screenshot = pyautogui.screenshot()
    before_screenshot.save(before_screenshot_path)
    
    # Execute the step
    step_fn()
    time.sleep(0.5)
    
    # Take screenshot AFTER executing the step
    after_screenshot_path = f"static/screenshots/{task_id}_step_{step_index + 1}.png"
    after_screenshot = pyautogui.screenshot()
    after_screenshot.save(after_screenshot_path)

    # Ask the agent if the step was successful using both before and after screenshots
    step_description = steps[step_index]
    success = ask_agent_for_verification_with_comparison(before_screenshot_path, after_screenshot_path, step_description)
    add_log(task_id, f"Step {step_index + 1} verification: {'Success' if success else 'Failed'}")
    
    not_success_counter = 0
    # If step failed, ask the LLM for a fix and retry
    while not success and not_success_counter < 5:
        add_log(task_id, f"Attempting to fix failed Step {step_index + 1}")
        fixed_function_code = ask_agent_to_fix_step_with_comparison(
            before_screenshot_path, after_screenshot_path, step_description, step_index
        )
        
        # Save the fixed function to a temporary file
        temp_file_path = f"fixed_step_{step_index+1}.py"
        with open(temp_file_path, "w") as f:
            f.write(fixed_function_code)
        
        # Import the fixed function
        temp_module_name = f"fixed_step_{step_index+1}"
        spec = importlib.util.spec_from_file_location(temp_module_name, temp_file_path)
        fixed_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixed_module)
        
        # Extract the fixed function
        fixed_fn_name = f"step_{step_index+1}"
        if hasattr(fixed_module, fixed_fn_name):
            fixed_fn = getattr(fixed_module, fixed_fn_name)
            
            # Take screenshot before executing the fixed function
            fixed_before_screenshot_path = f"static/screenshots/{task_id}_step_{step_index + 1}_fixed_before.png"
            fixed_before_screenshot = pyautogui.screenshot()
            fixed_before_screenshot.save(fixed_before_screenshot_path)
            
            # Try to execute the fixed function
            add_log(task_id, f"Executing fixed version of Step {step_index + 1}")
            fixed_fn()
            time.sleep(0.5)
            
            # Take new screenshot after executing the fixed function
            fixed_after_screenshot_path = f"static/screenshots/{task_id}_step_{step_index + 1}_fixed.png"
            fixed_after_screenshot = pyautogui.screenshot()
            fixed_after_screenshot.save(fixed_after_screenshot_path)
            
            # Verify if the fixed function worked using both before and after screenshots
            success = ask_agent_for_verification_with_comparison(
                fixed_before_screenshot_path, fixed_after_screenshot_path, step_description
            )
            add_log(task_id, f"Fixed Step {step_index + 1} verification: {'Success' if success else 'Failed'}")
            
            # Update the original steps module with the fixed function if successful
            if success:
                # Update steps_generated.py with the fixed function code
                with open(f"steps_generated_{task_id}.py", "r") as f:
                    existing_code = f.read()
                
                # Find and replace the specific function in the file
                import re
                pattern = rf"def {fixed_fn_name}\s*\(.*?\).*?(?=def|$)"
                replacement = fixed_function_code.strip() + "\n\n"
                updated_code = re.sub(pattern, replacement, existing_code, flags=re.DOTALL)
                
                with open(f"steps_generated_{task_id}.py", "w") as f:
                    f.write(updated_code)
        else:
            add_log(task_id, f"Error: Fixed function {fixed_fn_name} not found in generated code")

        not_success_counter += 1
    
    return success

def run_automation_thread(task_id, markdown_guide):
    try:
        add_log(task_id, "Starting automation...")
        
        # Extract steps from markdown guide
        steps = re.findall(r'\d+\.\s+(.*)', markdown_guide)
        
        add_log(task_id, f"Found {len(steps)} steps to execute")

        for step in steps:
        
            # Generate code for the steps
            code_str = clean_python_code(ask_agent(step))
            
            # Save generated functions to a file
            with open(f"steps_generated_{task_id}.py", "w") as f:
                f.write(code_str)
            
            # Import the dynamically created step functions
            spec = importlib.util.spec_from_file_location(f"steps_generated_{task_id}", f"steps_generated_{task_id}.py")
            steps_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(steps_module)
            
            # Get step functions
            step_functions = [
                getattr(steps_module, name)
                for name in dir(steps_module)
                if name.startswith("step_") and callable(getattr(steps_module, name))
            ]
            
            # Sort the steps by their number
            step_functions = sorted(step_functions, key=lambda fn: int(fn.__name__.split('_')[1]))
            
            # Execute steps
            for idx, step_fn in enumerate(step_functions):
                try:
                    add_log(task_id, f"Executing step {idx + 1} of {len(step_functions)}")
                    success = execute_step(task_id, step_fn, idx, steps)
                    if not success:
                        add_log(task_id, f"Step {idx + 1} verification failed even after attempting a fix. Stopping execution.")
                        break
                except Exception as e:
                    add_log(task_id, f"Error in step {idx + 1}: {str(e)}")
                    traceback.print_exc()
                    break
            
        add_log(task_id, "Automation sequence completed")
            
    except Exception as e:
        add_log(task_id, f"Error: {str(e)}")
        traceback.print_exc()
    finally:
        automation_tasks[task_id]['status'] = 'completed'

def add_log(task_id, message):
    if task_id in automation_tasks:
        automation_tasks[task_id]['logs'].append(message)
        print(f"Task {task_id}: {message}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start-automation', methods=['POST'])
def start_automation():
    markdown_guide = request.form.get('guide', '')
    
    if not markdown_guide:
        return jsonify({'status': 'error', 'message': 'No automation instructions provided'})
    
    # Generate a unique task ID
    task_id = str(int(time.time()))
    
    # Initialize task tracking
    automation_tasks[task_id] = {
        'status': 'running',
        'logs': [],
        'start_time': time.time()
    }
    
    # Start automation in a separate thread
    threading.Thread(target=run_automation_thread, args=(task_id, markdown_guide), daemon=True).start()
    
    return jsonify({'status': 'success', 'task_id': task_id})

@app.route('/check-status/<task_id>')
def check_status(task_id):
    if task_id not in automation_tasks:
        return jsonify({'status': 'error', 'message': 'Task not found'})
    
    task = automation_tasks[task_id]
    return jsonify({
        'status': task['status'],
        'logs': task['logs'],
        'elapsed_time': round(time.time() - task['start_time'])
    })

if __name__ == '__main__':
    # Clear the log file
    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write("")

    log_llm_output("gemini-2.5-pro", "Starting the web interface")
        
    # Run the Flask app
    app.run(debug=True)