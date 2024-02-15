import os, os.path

print(len([name for name in os.listdir('dataset/gum/gummy_cropped')]))
print(len([name for name in os.listdir('dataset/gum/normal_cropped')]))

print("ai_normal_cropped ", len([name for name in os.listdir('dataset/ai_gum/ai_normal_cropped')]))
print("ai_gummy_cropped ", len([name for name in os.listdir('dataset/ai_gum/ai_gummy_cropped')]))

print("mixed_gummy ", len([name for name in os.listdir('dataset/mixed_gum/mixed_gummy')]))
print("mixed_normal ", len([name for name in os.listdir('dataset/mixed_gum/mixed_normal')]))

print("ai_test ", len([name for name in os.listdir('dataset/ai_test')]))