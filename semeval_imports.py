import spacy
import nltk
import pickle
import math
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import fmin_l_bfgs_b as fmin
import os
import jsonrpc
from simplejson import loads
from nltk.tree import Tree
import codecs
import itertools
from itertools import product
from nltk.corpus import wordnet as wn
from gurobipy import *
from datetime import datetime
import copy
import random
from gensim.models import KeyedVectors

