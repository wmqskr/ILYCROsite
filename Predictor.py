import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import pandas as pd
import tkinter.font as tkFont
import os
from Bio import SeqIO
from MLP_Attention import MLP, Attention

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Model loading
def load_model(model_path):
    model = MLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model('./MLP_ATT_model6.pth')  # Specify model path

# GUI main program
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Lysine Crotonylation Site Prediction")
        font = tkFont.Font(family="Sans Serif", size=16)

        # Text box
        self.left_text = tk.Text(root, height=30, width=50)
        self.left_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
        self.left_text.configure(font=font)

        self.right_text = tk.Text(root, height=30, width=50)
        self.right_text.grid(row=0, column=2,columnspan=2, padx=10, pady=10)
        self.right_text.configure(font=font)

        # Buttons
        self.open_button = tk.Button(root, text="Select File", command=self.open_file)
        self.open_button.grid(row=1, column=0, sticky=tk.W+tk.E, padx=10, pady=10)

        self.predict_button = tk.Button(root, text="Predict", command=self.predic_from_csv)
        self.predict_button.grid(row=1, column=1, sticky=tk.W+tk.E, padx=10, pady=10)

        self.save_button = tk.Button(root, text="Save File", command=self.save_file)
        self.save_button.grid(row=1, column=2, sticky=tk.W+tk.E, padx=10, pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_windows)
        self.clear_button.grid(row=1, column=3, sticky=tk.W + tk.E, padx=10, pady=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                self.left_text.delete('1.0', tk.END)
                self.left_text.insert(tk.END, file.read())
                self.right_text.delete('1.0', tk.END)

            def read_fasta_and_process(file_name):
                # Reading the fasta file
                sequences = []
                for record in SeqIO.parse(file_name, "fasta"):
                    sequences.append(str(record.seq))

                # Removing the middle element 'T'
                n = len(sequences)
                m = (len(sequences[0]) + 1) // 2
                for i in range(n):
                    sequences[i] = sequences[i][:m - 1] + sequences[i][m:]  # remove the middle character

                return sequences, n

            def compute_ppt(sequences, n):
                AA = 'ACDEFGHIKLMNPQRSTVWY'
                PPT = pd.DataFrame(0, index=range(n), columns=list(AA))

                # Fill the PPT matrix
                for idx, seq in enumerate(sequences):
                    m = len(seq)
                    for char in seq:
                        if char in AA:
                            PPT.at[idx, char] += 1
                    PPT.iloc[idx] = PPT.iloc[idx] / m  # Normalize by the length of sequence

                return PPT
            positive_sequences, Np = read_fasta_and_process(file_path)
            PPT1 = compute_ppt(positive_sequences, Np)
            PPT1.dropna(how='all', inplace=True)
            PPT1.to_csv('testmany.csv', index=False, header=False)

    def predic_from_csv(self):
        file_path = "testmany.csv"
        if file_path:
            #df = pd.read_csv(file_path)
            df = pd.read_csv(file_path)
            if df.shape[1] != 20:
                messagebox.showerror("Error", "File content does not meet the requirements, please select again")
                return
            data = df.to_numpy()
            # Convert data to tensor and predict
            tensor = torch.tensor(data, dtype=torch.float32)
            output = model(tensor)
            predictions = ["Positive" if x > 0.5 else "Negative" for x in output.squeeze().tolist()]
            probabilities = [x if pred == "Positive" else 1-x for x, pred in zip(output.squeeze().tolist(), predictions)]
            results = [f"Sample {i+1}: {pred} (Prediction probability: {prob:.4f})" for i, (pred, prob) in enumerate(zip(predictions, probabilities))]
            self.right_text.delete('1.0', tk.END)
            self.right_text.insert(tk.END, "\n".join(results))

    def save_file(self):
        data = self.right_text.get('1.0', tk.END)
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(data)

    def clear_windows(self):
        os.remove("testmany.csv")
        self.left_text.delete('1.0', tk.END)
        self.right_text.delete('1.0', tk.END)
        messagebox.showinfo("Done", "Cleared")

# Create and start gui
root = tk.Tk()
app = App(root)
root.iconbitmap('protein.ico')
root.resizable(False, False)
root.mainloop()