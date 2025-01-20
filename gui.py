import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from threading import Thread
from prefect import flow, task
import logging
from typing import Optional
from api_client import APIClient
from intent_refiner import IntentRefiner
from output_handler import OutputHandler
import markdown2
import tkhtmlview
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModernTheme:
    DARK_BG = "#222831"     # Dark background
    DARKER_BG = "#393E46"   # Darker background for contrast
    ACCENT = "#00ADB5"      # Accent color for highlights
    TEXT = "#EEEEEE"        # Light text color
    COMBOBOX_BG = "#FFFFFF" # White background for combobox
    COMBOBOX_FG = "#000000" # Black text for combobox
    OUTPUT_BG = "#FFFFFF"   # White background for output
    OUTPUT_FG = "#000000"   # Black text for output

    @staticmethod
    def apply_theme(root):
        style = ttk.Style()
        style.theme_use('clam')  # Use clam as base theme

        # Configure colors for various widget states
        style.configure('Modern.TFrame', background=ModernTheme.DARK_BG)
        style.configure('Modern.TLabelframe', background=ModernTheme.DARK_BG, foreground=ModernTheme.TEXT)
        style.configure('Modern.TLabelframe.Label', background=ModernTheme.DARK_BG, foreground=ModernTheme.TEXT)
        style.configure('Modern.TLabel', background=ModernTheme.DARK_BG, foreground=ModernTheme.TEXT)
        style.configure('Modern.TButton',
            background=ModernTheme.ACCENT,
            foreground=ModernTheme.TEXT,
            padding=(20, 10),
            font=('Segoe UI', 10)
        )
        style.map('Modern.TButton',
            background=[('active', ModernTheme.DARKER_BG)],
            foreground=[('active', ModernTheme.TEXT)]
        )
        style.configure('Modern.TCombobox',
            background=ModernTheme.COMBOBOX_BG,
            foreground=ModernTheme.COMBOBOX_FG,
            fieldbackground=ModernTheme.COMBOBOX_BG,
            darkcolor=ModernTheme.ACCENT,
            lightcolor=ModernTheme.ACCENT,
            arrowcolor=ModernTheme.DARKER_BG
        )

        return style

# Initialize components
api_client = APIClient()
intent_refiner = IntentRefiner(api_client)
output_handler = OutputHandler()

@task
def refine_intent_task(initial_input: str, task: str, domain: str):
    return intent_refiner.refine_intent(initial_input, task, domain)

@task
def call_llm_task(prompt: str, model_type: str = None, task: str = None, domain: str = None, stream: bool = False):
    return api_client.call_llm_with_ensemble(prompt, task, domain)

@task
def format_output_task(output: str, format: str = "plain_text", domain: str = None):
    return output_handler.format_output(output, format, domain)

@flow(name="AI Agent Workflow")
def my_workflow(initial_input: str, task: str, domain: str, output_format: str = "plain_text"):
    refined_input = refine_intent_task(initial_input, task, domain)
    llm_result = call_llm_task(refined_input, task=task, domain=domain)
    formatted_output = format_output_task(llm_result, format=output_format, domain=domain)
    return formatted_output

def ask_question_dialog(question: str) -> Optional[str]:
    """Shows a question dialog in the output window and waits for user input"""
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"Clarifying Question:\n{question}\n\nPlease type your response below and click 'Submit Response'")
    output_text.config(state=tk.DISABLED)
    
    # Create a dialog for the response
    dialog = tk.Toplevel(root)
    dialog.title("Clarification Response")
    dialog.geometry("500x300")
    dialog.transient(root)
    dialog.grab_set()  # Make dialog modal
    
    # Response text area
    response_text = scrolledtext.ScrolledText(
        dialog,
        wrap=tk.WORD,
        height=10,
        bg=ModernTheme.DARKER_BG,
        fg=ModernTheme.TEXT,
        insertbackground=ModernTheme.ACCENT,
        font=('Segoe UI', 10)
    )
    response_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    response_var = {'response': None}
    
    def submit_response():
        response_var['response'] = response_text.get("1.0", tk.END).strip()
        dialog.destroy()
    
    def skip_response():
        response_var['response'] = None
        dialog.destroy()
    
    # Buttons
    button_frame = ttk.Frame(dialog, style='Modern.TFrame')
    button_frame.pack(fill=tk.X, padx=10, pady=10)
    
    submit_btn = ttk.Button(
        button_frame,
        text="Submit Response",
        command=submit_response,
        style='Modern.TButton'
    )
    submit_btn.pack(side=tk.LEFT, padx=5)
    
    skip_btn = ttk.Button(
        button_frame,
        text="Skip",
        command=skip_response,
        style='Modern.TButton'
    )
    skip_btn.pack(side=tk.LEFT, padx=5)
    
    # Wait for dialog to close
    dialog.wait_window()
    
    return response_var['response']

def run_workflow():
    initial_input = input_entry.get("1.0", tk.END).strip()
    task = task_combobox.get()
    domain = domain_combobox.get()
    output_format = output_format_combobox.get()

    if not initial_input:
        messagebox.showwarning("Missing Input", "Please enter your request.")
        return

    # Show processing indicator
    output_text.config(state=tk.NORMAL)
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, "Processing your request...\n")
    output_text.config(state=tk.DISABLED)
    
    # Disable only the main run button while processing, keep input fields enabled for clarification
    run_button.config(state=tk.DISABLED)

    def workflow_thread():
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(my_workflow, initial_input, task, domain, output_format)
                try:
                    result = future.result(timeout=300)  # 5-minute timeout
                    # Update display based on selected format
                    format_type = output_format_combobox.get()
                    update_output_display(result, format_type)
                except concurrent.futures.TimeoutError:
                    messagebox.showerror("Error", "The operation timed out. Please try again.")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n\nPlease check the logs for more details."
            messagebox.showerror("Error", error_message)
            logger.error(f"Workflow error: {e}", exc_info=True)
        finally:
            # Re-enable the run button
            run_button.config(state=tk.NORMAL)

    Thread(target=workflow_thread).start()

def set_ui_state(state):
    input_entry.config(state=state)
    task_combobox.config(state=state)
    domain_combobox.config(state=state)
    output_format_combobox.config(state=state)
    run_button.config(state=state)

def create_main_window():
    root = tk.Tk()
    root.title("Chun's long chain prompter")
    root.geometry("1200x800")
    root.minsize(800, 600)
    
    # Apply modern theme
    style = ModernTheme.apply_theme(root)
    root.configure(bg=ModernTheme.DARK_BG)

    # Create main container
    main_container = ttk.Frame(root, style='Modern.TFrame')
    main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Create left panel (input section)
    left_panel = ttk.LabelFrame(main_container, text="Input", style='Modern.TLabelframe')
    left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

    # Input widgets with modern styling
    input_label = ttk.Label(left_panel, text="Enter your request:", style='Modern.TLabel')
    input_label.pack(pady=(10, 5), padx=10, anchor='w')

    global input_entry
    input_entry = scrolledtext.ScrolledText(
        left_panel, 
        wrap=tk.WORD, 
        height=10,
        bg=ModernTheme.DARKER_BG,
        fg=ModernTheme.TEXT,
        insertbackground=ModernTheme.ACCENT,
        relief=tk.FLAT,
        font=('Segoe UI', 10)
    )
    input_entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # Task selection
    task_label = ttk.Label(left_panel, text="Task:", style='Modern.TLabel')
    task_label.pack(pady=(10, 5), padx=10, anchor='w')

    global task_combobox
    task_combobox = ttk.Combobox(
        left_panel,
        values=["text_generation", "question_answering", "data_analysis"],
        style='Modern.TCombobox',
        state='readonly'
    )
    task_combobox.pack(fill=tk.X, padx=10, pady=(0, 10))
    task_combobox.current(0)

    # Domain selection
    domain_label = ttk.Label(left_panel, text="Domain:", style='Modern.TLabel')
    domain_label.pack(pady=(10, 5), padx=10, anchor='w')

    global domain_combobox
    domain_combobox = ttk.Combobox(
        left_panel,
        values=["education", "marketing", "data_analysis"],
        style='Modern.TCombobox',
        state='readonly'
    )
    domain_combobox.pack(fill=tk.X, padx=10, pady=(0, 10))
    domain_combobox.current(0)

    # Output format selection
    format_label = ttk.Label(left_panel, text="Output Format:", style='Modern.TLabel')
    format_label.pack(pady=(10, 5), padx=10, anchor='w')

    global output_format_combobox
    output_formats = [
        ("Plain Text", "plain_text"),
        ("Markdown (Rich Text)", "markdown_rich"),
        ("Markdown (Raw)", "markdown_raw"),
        ("JSON", "json"),
        ("HTML", "html")
    ]
    
    output_format_combobox = ttk.Combobox(
        left_panel,
        values=[format[0] for format in output_formats],
        style='Modern.TCombobox',
        state='readonly'
    )
    output_format_combobox.pack(fill=tk.X, padx=10, pady=(0, 10))
    output_format_combobox.current(0)

    # Create run button
    global run_button
    run_button = ttk.Button(
        left_panel,
        text="Run",
        command=run_workflow,
        style='Modern.TButton'
    )
    run_button.pack(fill=tk.X, padx=10, pady=(10, 10))

    # Create export button
    export_button = ttk.Button(
        left_panel,
        text="Export Output",
        command=lambda: export_output(output_text.get("1.0", tk.END).strip()),
        style='Modern.TButton'
    )
    export_button.pack(fill=tk.X, padx=10, pady=(10, 10))

    # Create right panel (output section)
    right_panel = ttk.LabelFrame(main_container, text="Output", style='Modern.TLabelframe')
    right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

    # Replace ScrolledText with HTML viewer for rich text support
    global output_text
    output_text = tkhtmlview.HTMLScrolledText(
        right_panel,
        bg=ModernTheme.OUTPUT_BG,
        fg=ModernTheme.OUTPUT_FG,
        font=('Segoe UI', 10),
        html='<body style="background-color: #FFFFFF; color: #000000;">Welcome!</body>'
    )
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Add this after creating comboboxes to set their text color
    task_combobox.configure(foreground=ModernTheme.COMBOBOX_FG)
    domain_combobox.configure(foreground=ModernTheme.COMBOBOX_FG)
    output_format_combobox.configure(foreground=ModernTheme.COMBOBOX_FG)

    # Set up the question callback for the intent refiner
    intent_refiner.set_question_callback(ask_question_dialog)

    return root

def export_output(content):
    """Export the output content to a file"""
    file_types = [
        ('Text files', '*.txt'),
        ('Markdown files', '*.md'),
        ('HTML files', '*.html'),
        ('JSON files', '*.json'),
        ('All files', '*.*')
    ]
    
    filename = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=file_types,
        title="Export Output"
    )
    
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

def update_output_display(content, format_type):
    """Update the output display based on the selected format"""
    output_text.config(state=tk.NORMAL)
    
    if format_type == "markdown_rich":
        # Convert Markdown to HTML and display
        html_content = markdown2.markdown(content, extras=['fenced-code-blocks'])
        styled_html = f'''
        <body style="background-color: #FFFFFF; color: #000000; font-family: 'Segoe UI';">
            {html_content}
        </body>
        '''
        output_text.set_html(styled_html)
    else:
        # For other formats, display as plain text
        output_text.delete("1.0", tk.END)
        output_text.insert(tk.END, content)
        # Set text color for plain text display
        output_text.configure(fg=ModernTheme.OUTPUT_FG, bg=ModernTheme.OUTPUT_BG)
    
    output_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = create_main_window()
    root.mainloop()