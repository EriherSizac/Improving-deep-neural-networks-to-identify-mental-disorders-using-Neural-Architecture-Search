#Libraries
import os
import glob
import copy
import time

import numpy as np

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow import keras
import random

from math import floor

from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


from IPython.display import display

from collections import namedtuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import make_scorer
import math

from keras import backend as BKN

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.keras.utils import plot_model
import contextlib

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error


import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.message import EmailMessage

seed_value=1


"""# Parametros para entrenamiento"""

# Dataset
batch_size = 64  # adjust as needed
ratio = 2        # Adjust this as per your requirement
patch_size = 64  # adjust as needed
overlap = 0.1
# Directory where images are stored
directory1 = '/home/Super-IR/Thesis_code/Tesis_code/DIV2K_train_HR'
directory2 = '/home/Super-IR/Thesis_code/Tesis_code/DIV2K_valid_HR'
#Model params
epochs = 5
learning_rate=3e-04
epsilon=1e-07
weight_decay=1e-8


''' Util para enviarnos correos con alertas'''

def send_email_alert(subject, body, to_emails):
    from_email = "lleopol.do93@gmail.com"  # Replace with your email
    password = "glor rjpz emmm wgyw"  # Replace with your app password

    # Create the email content
    message = EmailMessage()
    message["From"] = from_email
    message["To"] = ", ".join(to_emails)
    message["Subject"] = subject
    message.set_content(body)

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Send email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(from_email, password)
            server.send_message(message)
        print("Emails sent successfully!")
    except Exception as e:
        print(f"Error sending emails: {e}")

"""# Utils para generar datasets de entrenamiento parcial"""

if os.path.exists("./tensor_y.npy"):
    os.remove("./tensor_y.npy")
if os.path.exists("./tensor_x.npy"):
    os.remove("./tensor_x.npy")
if os.path.exists("./val_tensor_x.npy"):
    os.remove("./val_tensor_x.npy")
if os.path.exists("./val_tensor_y.npy"):
    os.remove("./val_tensor_y.npy")

def image_generator(directory, batch_size, ratio, patch_size, overlap=0.5):
    filenames = os.listdir(directory)
    np.random.shuffle(filenames)
    for filename in filenames:
        img_path = os.path.join(directory, filename)
        image = load_img(img_path)
        image = img_to_array(image) / 255.0  # Normalize

        step_size = int(patch_size * (1 - overlap))
        batch_LR = []
        batch_HR = []

        for y in range(0, image.shape[0] - patch_size + 1, step_size):
            for x in range(0, image.shape[1] - patch_size + 1, step_size):
                patch_HR = image[y:y + patch_size, x:x + patch_size]
                patch_LR = tf.image.resize(patch_HR, (patch_size // ratio, patch_size // ratio), method='bicubic', antialias=True)

                patch_HR = np.clip(patch_HR, 0, 1)
                patch_LR = np.clip(patch_LR, 0, 1)

                batch_LR.append(patch_LR)
                batch_HR.append(patch_HR)

                if len(batch_LR) == batch_size:
                    yield np.array(batch_LR), np.array(batch_HR)
                    batch_LR = []
                    batch_HR = []
        if batch_LR:  # Check if there are remaining patches
            yield np.array(batch_LR), np.array(batch_HR)
            batch_LR, batch_HR = [], []

        # No handling for remaining patches that don't form a full batch

# Creating datasets
dataset_train = tf.data.Dataset.from_generator(
    lambda: image_generator(directory1, batch_size, ratio, patch_size, overlap),
    output_signature=(
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)
    )
).prefetch(tf.data.experimental.AUTOTUNE)  # Added prefetching for performance

dataset_val = tf.data.Dataset.from_generator(
    lambda: image_generator(directory2, batch_size, ratio, patch_size, overlap),
    output_signature=(
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.float32)
    )
).prefetch(tf.data.experimental.AUTOTUNE)

"""# EMOA"""

PRIMITIVES = [
    'conv', #tf.keras.layers.Conv2D#
    'dil_conv_d2', #tf.keras.layers.Conv2D dilation_rate = 2
    'dil_conv_d3', #tf.keras.layers.Conv2D dilation_rate = 3
    'dil_conv_d4', #tf.keras.layers.Conv2D dilation_rate = 4
    'Dsep_conv', #DepthwiseSeparableConv2D or tfm.vision.layers.DepthwiseSeparableConvBlock#
    'invert_Bot_Conv_E2', #tfm.vision.layers.InvertedBottleneckBlock#
    'conv_transpose', #tf.keras.layers.Conv2DTranspose#
    'identity' #tf.keras.layers.Identity
]

CHANNELS = [
    16,
    32,
    48,
    64,
    16,
    32,
    48,
    64
]

REPEAT = [
    1,
    2,
    3,
    4,
    1,
    2,
    3,
    4
]

K = [
    1,
    3,
    5,
    7,
    1,
    3,
    5,
    7
]

Genotype = namedtuple('Genotype', 'Branch1 Branch2 Branch3')

binary_dict = {}

def convert_cell(cell_bit_string):
    # convert cell bit-string to genome
    tmp = [cell_bit_string[i:i + 3] for i in range(0, len(cell_bit_string), 3)]
    return [tmp[i:i + 3] for i in range(0, len(tmp), 3)]


def convert(bit_string):
    # convert network bit-string to genome
    b1 = convert_cell(bit_string[:len(bit_string)//3])
    b2 = convert_cell(bit_string[len(bit_string)//3:(len(bit_string)//3)*2])
    b3 = convert_cell(bit_string[(len(bit_string)//3)*2:])
    return [b1, b2, b3]

def decode(genome):
    genotype = copy.deepcopy(genome)
    channels = genome.pop(0)
    genotype = convert(genome)
    # decodes genome to architecture
    b1 = genotype[0]
    b2 = genotype[1]
    b3 = genotype[2]

    branch1, branch1_concat = [('channels', CHANNELS[channels])], list(range(2, len(b1)+2))
    branch2, branch2_concat = [('channels', CHANNELS[channels])], list(range(2, len(b2)+2))
    branch3, branch3_concat = [('channels', CHANNELS[channels])], list(range(2, len(b3)+2))

    for block in b1:
        for unit in block:
            branch1.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))

    for block in b2:
        for unit in block:
            branch2.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))
    for block in b3:
        for unit in block:
            branch3.append((PRIMITIVES[unit[0]], [K[unit[1]],K[unit[1]]], REPEAT[unit[2]]))

    #print(Genotype(Branch1=branch1,Branch2=branch2,Branch3=branch3))
    return Genotype(
        Branch1=branch1,
        Branch2=branch2,
        Branch3=branch3
    )

def get_branches(genotype):
  gens = copy.deepcopy(genotype)
  conv_args = {
      "activation": "relu",
      "padding": "same",
      }
  channels = []
  for element in gens:
    channels.append(element.pop(0))
  branches = [[],[],[]]

  for i in range(len(gens)):

    for layer in gens[i]:

      if layer[0] == 'conv':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], **conv_args)])

      elif layer[0] == 'dil_conv_d2':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=2,**conv_args)])

      elif layer[0] == 'dil_conv_d3':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=3,**conv_args)])

      elif layer[0] == 'dil_conv_d4':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(channels[i][1], layer[1], dilation_rate=4,**conv_args)])

      elif layer[0] == 'Dsep_conv':
        for l in range(layer[2]):
          branches[i].extend([layers.DepthwiseConv2D(layer[1], **conv_args), layers.Conv2D(channels[i][1], 1, **conv_args)])

      elif layer[0] == 'invert_Bot_Conv_E2':
        expand = float(channels[i][1])*2
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2D(expand, 1,**conv_args), layers.DepthwiseConv2D(layer[1], **conv_args),layers.Conv2D(channels[i][1], layer[1],**conv_args)])

      elif layer[0] == 'conv_transpose':
        for l in range(layer[2]):
          branches[i].extend([layers.Conv2DTranspose(channels[i][1], layer[1], **conv_args)])

      elif layer[0] == 'identity':
          branches[i].extend([layers.Identity()])

      else:
        print("what?", i, layer[0])
      #  branches[i].extend(block)
  bc = []
  bc.extend(branches)
  bc.append(channels[0][1])
  return bc

def gray_to_int(gray_code):
    # Convert the Gray code string to a list of integers (0 or 1)
    gray_bits = [int(bit) for bit in gray_code]

    # The first bit of the Gray code is the same as the first bit of the binary code
    binary_bits = [gray_bits[0]]

    # Compute the binary code bits from the Gray code bits
    for i in range(1, len(gray_bits)):
        # The next binary bit is the XOR of the next Gray bit and the previous binary bit
        next_binary_bit = gray_bits[i] ^ binary_bits[i-1]
        binary_bits.append(next_binary_bit)

    # Combine the binary bits into a single integer
    binary_str = ''.join(str(bit) for bit in binary_bits)
    integer = int(binary_str, 2)

    return integer

def bstr_to_rstr(bstring):
  rstr = []
  for i in range(0,len(bstring),3):
    r = gray_to_int(bstring[i:i+3])
    rstr.append(r)
  return(rstr)

def get_model(genotype, upscale_factor=2, channels = 3):

  branch1, branch2, branch3, channels_mod = get_branches(genotype)

  conv_args = {
        "activation": "relu",
        "padding": "same",
    }

  inputs = layers.Input(shape=(None, None, channels), dtype='float32')

  inp = layers.Conv2D(channels_mod, 3, **conv_args)(inputs)

  b1 = branch1[0](inp)

  for l in range(1,len(branch1)):
    b1 = branch1[l](b1)

  b2 = branch2[0](inp)

  for l in range(1,len(branch2)):
    b2 = branch2[l](b2)

  b3 = branch3[0](inp)

  for l in range(1,len(branch3)):
    b3 = branch3[l](b3)

  x = layers.Add()([b1,b2,b3])
  #x = layers.Conv2D(12, 3, **conv_args)(b3)
  x = layers.Conv2D(12, 3, **conv_args)(x)
  x = tf.nn.depth_to_space(x, upscale_factor)
  outputs = layers.Conv2D(3, 3, **conv_args, dtype='float32')(x)
  
  return keras.Model(inputs, outputs)

class ReferencePoint:
    def __init__(self, position):
        self.position = position

# NSGA-III implementation in python


class NSGA3:
    def __init__(self, pop_size=100, n_gen=1000, problem=None, verbose=False):
        """
        Initialize the NSGA-III optimizer.
        :param pop_size: Population size.
        :param n_gen: Number of generations.
        :param problem: The optimization problem to be solved.
        :param verbose: If True, print verbose output.
        """
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.verbose = verbose
        self.n_eval = 1  # Initialize evaluation counter

    @staticmethod
    def _dominate(p, q):
        """
        Check if one solution dominates another.
        :param p: The first solution.
        :param q: The second solution.
        :return: True if p dominates q, False otherwise.
        """
        return all(p_i < q_i for p_i, q_i in zip(p, q))

    def _fast_non_dominated_sorting(self, pop):
        """
        Perform fast non-dominated sorting on the population.
        :param pop: Population to be sorted.
        :return: Sorted fronts of the population.
        """
        f1 = []
        fronts = {}
        sp = {}
        nq = {}
        for i in range(len(pop['F'])):
            p = pop['F'][i]
            s = []
            n = 0
            for j in range(len(pop['F'])):
                q = pop['F'][j]
                if self._dominate(p, q):
                    s.append(j)
                elif self._dominate(q, p):
                    n += 1
            sp[f'p_{i}'] = s
            nq[f'q_{i}'] = n
            if n == 0:
                f1.append(i)
        k = 1
        fronts[f'F{k}'] = f1
        while fronts[f'F{k}'] != []:
            next_front = []
            for i in fronts[f'F{k}']:
                for j in sp[f'p_{i}']:
                    nq[f'q_{j}'] -= 1
                    if nq[f'q_{j}'] == 0:
                        next_front.append(j)
            k += 1
            fronts[f'F{k}'] = next_front
        return fronts

    def _initialize_pop(self):
        """
        Initialize the population.
        :return: Initialized population and number of evaluations.
        """
        x, x_f = [], []
        for _ in range(self.pop_size):
            ind = []
            for i in range(self.problem.n_var):
                value = np.random.randint(0, 2)
                ind.append(value)
            x.append(ind)
            x_f.append(self.problem._evaluate_multi(ind, self.n_eval))
            self.n_eval += 1
        return {'X': x, 'F': x_f}

    def _new_individual(self, pop):
        """
        Generate a new individual.
        :param pop: Current population.
        :return: New individual and its fitness.
        """
        parents = TournamentSelection(n_parents=2)(pop=pop)
        offspring = KPointBinaryCrossover(problem=self.problem)(parents, pop)
        x = BitFlipMutation(problem=self.problem, prob=1 / self.problem.n_var)(offspring)
        x_f = self.problem._evaluate_multi(x, self.n_eval)
        self.n_eval += 1
        return {'X': [list(x)], 'F': [x_f]}

    def _generate_q(self, pop):
        """
        Generate a new population Q.
        :param pop: Current population P.
        :return: New population Q.
        """
        x = []
        x_f = []
        for _ in range(self.pop_size):
            q = self._new_individual(pop)
            x.append(q['X'][0])
            x_f.append(q['F'][0])
        return {'X': x, 'F': x_f}


    def generate_reference_points(self, num_objs, num_divisions_per_obj= 4):
        def gen_refs_recursive(work_point, num_objs, left, total, depth):
            if depth == num_objs - 1:
                work_point[depth] = left/total
                ref = ReferencePoint(copy.deepcopy(work_point))
                return [ref]
            else:
                res = []
                for i in range(left + 1):
                    work_point[depth] = i/total
                    res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
                return res

        return gen_refs_recursive([0]*num_objs, num_objs, num_objs*num_divisions_per_obj,
                                  num_objs*num_divisions_per_obj, 0)


    def _weights_vector(self):
        # Generates the weights vector using reference points
        reference_points = self.generate_reference_points(num_objs=self.problem.n_obj)
        w_v = {f'w{i}': ref.position for i, ref in enumerate(reference_points)}
        return w_v

    def _normalize(self, s, pop):
        min_obj = [np.inf] * self.problem.n_obj
        max_obj = [-np.inf] * self.problem.n_obj

        # Ensure indices in 's' are within the range of 'pop'
        valid_s = [index for index in s if 0 <= index < len(pop)]

        for index in valid_s:
            for obj in range(self.problem.n_obj):
                # Additional check for 'obj' range in 'pop[index]'
                if obj < len(pop[index]):
                    if pop[index][obj] < min_obj[obj]:
                        min_obj[obj] = pop[index][obj]
                    if pop[index][obj] > max_obj[obj]:
                        max_obj[obj] = pop[index][obj]
                else:
                    print(f"Obj index {obj} is out of range for the population at index {index}.")

        # Normalize the values of the population
        normalized = []
        for index in valid_s:
            ind = [(pop[index][obj] - min_obj[obj]) / (max_obj[obj] - min_obj[obj]) if max_obj[obj] != min_obj[obj] else 0 for obj in range(self.problem.n_obj)]
            ind.append(index)
            normalized.append(ind)

        return normalized


    # Change done to generalize to N dimensions.

    def _associate(self, norm):
        a = {}
        for index in range(len(norm)):
            # Adjust to include all objectives
            ind = norm[index][:self.problem.n_obj]
            d_min = np.inf
            w_min = []
            for w_index in range(len(self.ref_points)):
                w = self.ref_points[f'w{w_index}']
                d = self.perpendicular_distance(w, ind)
                if d < d_min:
                    d_min = d
                    w_min = f'w{w_index}'
            a[f'{norm[index][self.problem.n_obj]}'] = [d_min, w_min]

        return a

    def perpendicular_distance(self, direction, point):    
        if len(direction) != len(point):
            print(f"Vector length mismatch: direction {len(direction)}, point {len(point)}")
            return np.inf
        print(np.dot(direction, point))
        k = np.dot(direction, point) / np.sum(np.power(direction, 2))
        d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point), 2))
        return np.sqrt(d)


    def _niching(self, l_front, niche, a, pop_index):
        k = self.pop_size - len(pop_index)
        count = 0
        flag = True
        while flag:
            l_ref = list(np.unique([a[f'{index}'][1] for index in l_front]))
            empty_ref = [key for key in niche.keys() if niche[key] == 0]
            if len(empty_ref) > 0:
                for w in empty_ref:
                    if w in l_ref:
                        d_min = np.inf
                        closer = None
                        for i in l_front:
                            if a[f'{i}'][1] == w:
                                if a[f'{i}'][0] < d_min:
                                    d_min = a[f'{i}'][0]
                                    closer = i
                        pop_index.append(closer)
                        l_front.remove(closer)
                        niche[w] += 1
                        count += 1
                    else:
                        niche.pop(w)
                    empty_ref.remove(w)

                    if count >= k:
                        flag = False
                        break
            else:
                for w in l_ref:
                    d_min = np.inf
                    closer = None
                    for i in l_front:
                        if a[f'{i}'][1] == w:
                            if a[f'{i}'][0] < d_min:
                                d_min = a[f'{i}'][0]
                                closer = i
                    pop_index.append(closer)
                    l_front.remove(closer)
                    niche[w] += 1
                    count += 1

                    if count >= k:
                        flag = False
                        break

        return pop_index

    def _non_dominated_samples(self, front):
        indexes = []
        for i in range(len(front['F'])):
            p = front['F'][i]
            n = 0
            for j in range(len(front['F'])):
                q = front['F'][j]
                if self._dominate(q, p):
                    n += 1
            if n == 0:
                indexes.append(i)

        return indexes

    def _do(self):
        """
        Execute the NSGA-III algorithm.
        :return: Final population and non-dominated solutions.
        """
        c = 0
        nds = {}
        self.ref_points = self._weights_vector()
        pop = self._initialize_pop()
        filename = f"XGBOOST_Output_Population_seed_{seed_value}.txt"
        with open (filename,"a") as file:
            for x,f in zip(pop["X"],pop["F"]):
                file.write(f"{x}" + f"{f}" + "\n" )

        # Initialize progress bar
        if self.verbose:
            pbar = tqdm(total=self.n_gen, desc='NSGA-III Progress')

        while c < self.n_gen:
            #if self.verbose:
            #    print(f"Generation {c}: Evaluating...")

            q_pop = self._generate_q(pop)

            #print(c)
            
            # Union between P and Q
            r_pop = {}
            for key in pop.keys():
                r_pop[key] = pop[key] + q_pop[key]

            # Fast-non-dominated sorting
            fronts = self._fast_non_dominated_sorting(r_pop)

            # Select the first l fronts until the size of s is equal or bigger than the pop_size
            s, f = [], 0
            while len(s) < self.pop_size:
                f += 1
                for x in fronts[f'F{f}']:
                    s.append(x)

            # Review the size of s to determine the next population
            if len(s) == self.pop_size:
                for key in r_pop.keys():
                    pop[key] = [r_pop[key][index] for index in s]
            else:
                pop_index = [item for j in range(1, f) for item in fronts[f'F{j}']]
                last_front = fronts[f'F{f}']

                # Normalize the s individuals objective values
                normal = self._normalize(s, r_pop['F'])

                # Associate each member of s with a reference point
                a = self._associate(normal)

                # Compute niche count of reference point
                niche_c = {}
                for key in self.ref_points:
                    niche_c[key] = 0
                for index in pop_index:
                    niche_c[a[f'{index}'][1]] += 1

                # Determine the pop_size elements for the new population
                pop_index = self._niching(last_front, niche_c, a, pop_index)

                for key in pop.keys():
                    pop[key] = [r_pop[key][i] for i in pop_index]

            # Obtain the nds list
            nds_index = self._non_dominated_samples(pop)

            # Determine the nds per population
            for key in pop.keys():
                nds[key] = [pop[key][index] for index in nds_index]

            filename = f"XGBOOST_Output_Population_seed_{seed_value}.txt"
            with open (filename,"a") as file:
                for x,f in zip(pop["X"],pop["F"]):
                    file.write(f"{x}" + f"{f}" + "\n" )
            c += 1
            pbar.update(1)  # Update progress bar
            
        if self.verbose:
            pbar.close()
        return pop, nds

    def __call__(self):
        return self._do()

class TournamentSelection:
    def __init__(self,
                 n_select=1,
                 n_parents=1,
                 pressure=2):
        self.n_select = n_select  # number of tournaments
        self.n_parents = n_parents  # number of parents
        self.pressure = pressure  # rate of convergence
        self.n_random = n_select * n_parents * pressure  # number of random individuals needed

    def random_permutations(self, length, concat=True):
        p = []
        for i in range(self.n_perms):
            p.append(np.random.permutation(length))
        if concat:
            p = np.concatenate(p)  # from matrix to vector
        return p

    def _do(self, pop):

        # number of permutations needed
        self.n_perms = math.ceil(self.n_random / len(pop['X']))

        # get random permutations and reshape them
        p = self.random_permutations(len(pop['X']))[:self.n_random]
        p = np.reshape(p, (self.n_select * self.n_parents, self.pressure))

        # compare using tournament function
        n_tournaments, _ = p.shape
        s = np.full(n_tournaments, np.nan)

        for i in range(n_tournaments):
            a, b = p[i, 0], p[i, 1]
            a_f, b_f = pop['F'][a], pop['F'][b]

            # if one dominates another choose the nds one
            rel = Dominance(a_f, b_f)
            if rel == 1:
                s[i] = a
            elif rel == -1:
                s[i] = b

            # if rank or domination relation didn't make a decision compare by crowding
            if np.isnan(s[i]):
                s[i] = np.random.choice([a, b])

        return s[:, None].astype(int, copy=False)

    def __call__(self, pop):
        return self._do(pop)

class KPointBinaryCrossover:

    def __init__(self, problem, k=2, prob=0.9):
        """
        Initialize the k-point binary crossover.
        :param problem: The optimization problem context.
        :param k: The number of crossover points.
        :param prob: The probability of crossover.
        """
        self.problem = problem
        self.k = k
        self.prob = prob

    def _crossover_points(self):
        """
        Generates crossover points.
        :return: A sorted list of unique crossover points.
        """
        points = np.random.choice(range(1, self.problem.n_var), self.k, replace=False)
        return sorted(points)

    def _repair(self, x):
        """
        Repairs the offspring if it is out of the problem bounds.
        :param x: The offspring to repair.
        :return: Repaired offspring.
        """
        for i in range(len(x)):
            if x[i] < self.problem.xl[i]:
                x[i] = self.problem.xl[i]
            elif x[i] > self.problem.xu[i]:
                x[i] = self.problem.xu[i]
        return x

    def _do(self, i_par, pop):
        """
        Perform the k-point crossover operation.
        :param i_par: Indices of the parents.
        :param pop: Population.
        :return: Offspring.
        """
        if np.random.rand() <= self.prob:
            parent1 = pop['X'][i_par[0][0]]
            parent2 = pop['X'][i_par[1][0]]
            off = np.full(self.problem.n_var, np.nan)

            crossover_points = self._crossover_points()
            crossover_points.append(self.problem.n_var)  # Ensures the last segment is included

            start = 0
            for i, point in enumerate(crossover_points):
                off[start:point] = parent1[start:point] if i % 2 == 0 else parent2[start:point]

                start = point

            off = self._repair(off)
        else:
            # If crossover does not occur, randomly select one of the parents to be the offspring
            off = pop['X'][np.random.choice([i_par[0][0], i_par[1][0]])]

        return off

    def __call__(self, parents, pop):
        """
        Callable interface for the crossover operation.
        :param parents: Indices of the parents.
        :param pop: Population.
        :return: Offspring.
        """
        return self._do(i_par=parents, pop=pop)

class BitFlipMutation:

    def __init__(self, problem, prob=0.1):
        """
        Initialize the bitflip mutation.
        :param problem: The optimization problem context.
        :param prob: The probability of each bit being flipped.
        """
        self.problem = problem
        self.prob = prob

    def _flip(self):
        """
        Generates flip decisions for each variable.
        :return: A list indicating if each variable should be flipped.
        """
        return np.random.rand(self.problem.n_var) < self.prob

    def _do(self, parent):
        """
        Perform the bitflip mutation on a parent.
        :param parent: The parent to mutate.
        :return: Mutated offspring.
        """
        flip_decision = self._flip()
        off = np.copy(parent)

        for i in range(self.problem.n_var):
            if flip_decision[i]:
                # Flip the bit/value
                if off[i] == 0:
                  off[i] = 1
                elif off[i] == 1:
                  off[i] = 0

        return off

    def __call__(self, parent):
        """
        Callable interface for the mutation operation.
        :param parent: The parent to mutate.
        :return: Mutated offspring.
        """
        return self._do(parent)

def Dominance(a_f, b_f):
    """
    Determines the weak Pareto dominance relationship between two vectors.
    Returns 1 if a_f weakly dominates b_f, -1 if b_f dominates a_f, and 0 if no dominance relation exists.

    :param a_f: List or array-like representing vector a_f.
    :param b_f: List or array-like representing vector b_f.
    :return: int - 1, 0, or -1 based on the dominance relationship.
    """
    a_dominates = False
    b_dominates = False

    for a, b in zip(a_f, b_f):
        if a > b:
            a_dominates = True
        elif b > a:
            b_dominates = True

        if a_dominates and b_dominates:
            # No dominance relation exists
            return 0

    if a_dominates:
        return 1
    elif b_dominates:
        return -1
    else:
        # No dominance, but vectors are identical
        return 0

"""# Objetivos a Optimizar"""

def psnr(orig, pred):
    # Scale and cast the target images to integer
    orig = tf.cast(orig * 255.0, tf.uint8)
    # Scale and cast the predicted images to integer
    pred = tf.cast(pred * 255.0, tf.uint8)
    # Return the PSNR
    return tf.image.psnr(orig, pred, max_val=255)

def calculate_model_flops(model):
    # Ensure the model is built with a defined input shape
    if not model.built:
        sample_input = tf.keras.Input(shape=model.input_shape[1:])
        model(sample_input)

    # Define the forward pass for the model
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    # TensorFlow Profiler gets FLOPs
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())

    # Divide by 2 as `profile` counts multiply and accumulate as two FLOPs
    flops = graph_info.total_float_ops // 2
    return flops

class ModifiedMyPMOP:
    def __init__(self, n_var=84, n_obj=3):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.zeros(self.n_var)  # Lower bounds
        self.xu = np.ones(self.n_var)  # Upper bounds
        self.boost = None
        self.train_xgboost()  # Automatically train the XGBoost model during initialization

    def train_xgboost(self):
        # Data is in a CSV file
        file_path = '/workspace/model_psnr_result.csv'
        data = pd.read_csv(file_path, header=None)

        # Extract features and target variables
        X = np.array([eval(row[0]) for row in data.values])
        y = data.iloc[:, -1]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.036, random_state=1)

        # Create XGBoost regression model
        xgb_model = xgb.XGBRegressor()
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15]
        }

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        # Best Params
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)
        self.boost = XGBRegressor(**best_params)
        self.boost.fit(X_train, y_train)

    def func_eval_model(self, ind, n_eval):
        ind = bstr_to_rstr(ind)
        x_ind = [ind]
        
        y_pred = self.boost.predict(x_ind)

        valid_psnr = y_pred[0]
        print(valid_psnr)
        
        return valid_psnr

    def func_eval_flops(self, ind):
        genotype = decode(bstr_to_rstr(ind))
        model = get_model(genotype)

        # Ensure the model is built, required for input shape
        if not model.built:
            model.build(input_shape=(None,) + model.input_shape[1:])

        # Define the forward pass for the model
        @tf.function
        def forward_pass(inputs):
            return model(inputs)

        # Convert Keras model to ConcreteFunction
        concrete_func = forward_pass.get_concrete_function(
            tf.TensorSpec(shape=(1, 64, 64, model.input_shape[-1]), dtype=tf.float32))

        # Get the frozen ConcreteFunction
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)

        # Suppress the profiler output
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            # Profile the model
            graph_info = profile(frozen_func.graph, options=ProfileOptionBuilder.float_operation())

        # Get total float operations (FLOPs)
        flops = graph_info.total_float_ops // 2  # //2 because profile counts multiply and accumulate as two flops
        return flops

    def func_eval_params(self, ind, random_seed=1):
        random.seed(random_seed)

        ind = bstr_to_rstr(ind)
        genotype = decode(ind)
        model = get_model(genotype)

        params = model.count_params()

        return params

    def _evaluate_multi(self, ind, n_eval):
        f1 = self.func_eval_model(ind, n_eval)
        f2 = self.func_eval_params(ind)
        f3 = self.func_eval_flops(ind)
        return [-f1, f2, f3]
    

print(f"Setting seed to: {seed_value}")
    

random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
    # Example usage
problem = ModifiedMyPMOP()

print(problem)
nsga3 = NSGA3(pop_size=20, n_gen=1250, problem=problem, verbose= True)

    # Run the algorithm
final_population, non_dominated_solutions = nsga3()

print(final_population)

print(non_dominated_solutions)

