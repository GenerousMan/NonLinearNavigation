import os, sys

sys.path.append(os.getcwd())

from classes.text import Text

text = Text("Decathlon Adults-Men Men Black Fashion Fall 2010 Sports Football Sports Shoes Footwear Shoes. Optimum stud configuration for added stability and grip 2. Reinforced heel and central lacing for good foot support 3. Cushioned, flexible outsole with flex grooves for shock absorption 4. Rubber outsole, EVA midsole, synthetic upper Designed for men who play football on firm ground. Rush now and order a pair for yourself before your next big soccer match is on.")
print(text.sentences)
print(text.tokenized)