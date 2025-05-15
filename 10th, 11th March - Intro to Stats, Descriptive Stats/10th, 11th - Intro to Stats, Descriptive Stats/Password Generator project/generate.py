import tkinter as tk 

# function to called when button is clicked
def on_button_click():
    Label.config(text = 'Button clicked')

#create the main application window 
root = tk.Tk()
root.title('Simple tkinter app')

#create label widget
Label = tk.Label(root, text ='hello Tkinter')
Label.pack(pady=20)

# button creation 
button = tk.Button(root, text = 'CLck me ', command = on_button_click)
button.pack(pady=20)

root.mainloop()