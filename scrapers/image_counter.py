import os, os.path

print(len([name for name in os.listdir('raw_data/gum/gummy')]))
print(len([name for name in os.listdir('dataset/gum/gummy_cropped')]))

print(len([name for name in os.listdir('raw_data/gum/normal')]))
print(len([name for name in os.listdir('dataset/gum/normal_cropped')]))
