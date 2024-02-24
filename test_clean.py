### import libraries
import numpy as np
import pandas as pd
import torch
import json
import regex as re
import clean_txt as clean

# Example text
text = "A man is in the box"#"hello world python"

text = clean.clean_txt(text, w2v = True)

print([word for word  in text.split()])