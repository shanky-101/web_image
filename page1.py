import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
#from torch.utils.data import DataLoader, random_split
import pandas as pd
import os 

def read_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)

            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(part.split('/')[0]) for part in parts[1:]]
                faces.append(face)

    return {'vertices': vertices, 'faces': faces}

def write_obj_verts(obj_file_path, new_vertices, output_txt_path):
    # Read the obj file
    with open(obj_file_path, 'r') as file:
        lines = file.readlines()

    updated_lines = []
    vertex_index = 0
    for line in lines:
        if line.startswith('v '):
            # Replace the vertex with the new vertex
            new_vertex = new_vertices[vertex_index]
            updated_line = f'v {new_vertex[0]} {new_vertex[1]} {new_vertex[2]}\n'
            updated_lines.append(updated_line)
            vertex_index += 1
        else:
            # Keep the other lines unchanged
            updated_lines.append(line)

    # Write the updated content to the output txt file
    with open(output_txt_path, 'w') as file:
        file.writelines(updated_lines)

    print(f'Updated vertices and copied contents to {output_txt_path}')

class NN(nn.Module):
    def __init__(self,input_size,output):
        super(NN,self).__init__()
        self.fc1=nn.Linear(input_size,300)
        self.fc2=nn.Linear(300,3000)
        self.fc3=nn.Linear(3000,output)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in',
                                 nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in',
                                 nonlinearity='relu')
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self,x):
        x=f.relu(self.fc1(x))
        x=f.relu(self.fc2(x))
        x=self.fc3(x)
        return x
# curr<ent_dir = os.getcwd()
script_path = os.path.dirname(__file__)

print(script_path)

#headless smplx f body
# file_path = os.path.join(current_dir, file_name)
temp_mesh_path = os.path.join(script_path,"reg_bodies_0_quad.objres.obj" )#"../reg_bodies_0_quad.objres.obj"
temp_mesh = read_obj(temp_mesh_path)
temp_verts = temp_mesh["vertices"]
num_verts = len(temp_verts)
faces = temp_mesh["faces"]
#print(len(temp_mesh_vertices))

input_size=12 #14 #number of dimensions +height +weight
output= int(9492*3) #number of body verticesx3 (9492x3)
batch_size= 4
#print(output)

csv_file_path = os.path.join(script_path, "male_reg_train_v4.csv")
df = pd.read_csv(csv_file_path)
df.drop(['Objname','Betas','height','weight'],axis = 1,inplace = True)
#df.head()
# Initialize lists to store max and min values of each dim
max_list = []
min_list = []
for column in df.columns:
    max_list.append(df[column].max())
    min_list.append(df[column].min())

# Display the results
#print("Max values for each column:", max_list)
#print("Min values for each column:", min_list)

#ouput normalization values
min_x = -0.1742
range_x = 0.3505
min_y = -0.0023
range_y = 0.3640
min_z = -0.0457
range_z = 0.1141

# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
#device
model_path = os.path.join(script_path,"Model/male_bodynet_v4_100_nowgt.pth")
#"/home/vikas/Desktop/Test_cont_pipeline/Male/1_Cont_headless_body/Model/male_bodynet_50.pth"
#/male_bodynet_20_nowgt.pth"
model_loaded = NN(input_size=input_size,output = output)
model_loaded.load_state_dict(torch.load(model_path,map_location=torch.device('cuda')))
model_loaded.eval()

def comma_separated_floats(value):
    """Parse a comma-separated string into a list of floats."""
    try:
        return [float(x) for x in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid input: {value}. Ensure it is a comma-separated list of numbers.")


#Test custom dimensions
import argparse
#import ast
parser = argparse.ArgumentParser(description="Enter body dimensions")
parser.add_argument(
        '--dim', 
        type=comma_separated_floats, 
        nargs='+',  # Allows multiple arguments
        required=True,  # Makes the argument mandatory
        help="A list of float numbers separated by spaces."
    )
parser.add_argument('--height', type=float, required=True, help="Height in cm")
parser.add_argument('--weight', type=float, required=True, help="Weight in kg")
#parser.add_argument('--size', type=float, required=True, help="Tshirt size")
args = parser.parse_args()
dim_list = args.dim
height = args.height          
weight = args.weight
#size = args.size
part2 = int(height)
part3 = int(weight) 
#size = int(size)

#custom dimesions
b = dim_list[0]
print("body_dimensions")
print(b)
inp = torch.zeros(input_size)
#check if in range
for i in range(input_size):
  if b[i] < max_list[i] or b[i] > min_list[i]:
    inp[i] = b[i]/max_list[i]
  else:
    print("Dimension not in range",i)
    print(b[i])
    print(max_list[i], min_list[i])
    break

#normalize inputs
#inp = a[0]
inp = torch.zeros(input_size)
for i in range(input_size):
  inp[i] = b[i]/max_list[i]
inp = torch.FloatTensor(inp)
# print(inp)
# print(inp.dtype)
# print(b.dtype)

#infer
inp = torch.FloatTensor(inp)
inp = inp.to(device = device)
model_loaded = model_loaded.to(device=device)
pred_disp = model_loaded(inp)
pred_disp = pred_disp.detach().cpu().numpy()
pred_disp = np.reshape(pred_disp, (num_verts,3))
#print(pred_disp)
# print(out)

outputs = np.zeros((num_verts,3))

#normalize output
for i in range(num_verts):
  ##x = (xnormalized * range of x) + xmin
  pred_disp[i][0] = (pred_disp[i][0] *range_x) + min_x
  pred_disp[i][1] = (pred_disp[i][1] *range_y) + min_y
  pred_disp[i][2] = (pred_disp[i][2] *range_z) + min_z
  outputs[i] = pred_disp[i] + temp_verts[i]

#save output
#height = int(b[-2])
#weight = int(b[-1])
#print(height,weight)
output_headless_dir = os.path.join(script_path,"Output_headless_bodies")
output_headless_name = f"custom_headless_{part2}_{part3}.obj"
output_headless_path = os.path.join(output_headless_dir,output_headless_name)
write_obj_verts(temp_mesh_path,outputs,output_headless_path)

#display
#pred_body_mesh = trimesh.load(f"custom_headless_{height}_{weight}.obj", process=False)
#pred_body_mesh.show()