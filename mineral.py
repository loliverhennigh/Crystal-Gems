
from io import StringIO, BytesIO
import requests
from lxml import etree
import os
import sys
import imghdr
import tensorflow as tf
import numpy as np
import cv2

COLORS = ["gray", "black", "white", "green", "yellow", "blue", "pink", "orange", "red", "colorless", "beige", "brown", "purple", "multicolored", "steel", "silver", "pale", "cream", "tin"]
STREAKS = ["gray", "black", "white", "green", "yellow", "blue", "pink", "orange", "red", "colorless", "beige", "brown", "purple", "multicolored", "steel", "silver", "pale", "cream", "tin"]
CRYSTAL_SYSTEM = ['hexagonal', 'monoclinic', 'tetragonal', 'isometric', 'triclinic', 'orthorhombic', 'amorphous']
LUSTERS = ["adamantine", "resinous"]
FRACTURES = ["subconchoidal", "splintery", "conchoidal", "uneven", "hackly", "cleavage", "earthy", "even"]
TENACITYS = ["brittle", "malleable", "sectile", "fibrous", "elastic", "inelastic", "ductile", "flexible", "nonbrittle"]
GROUPS = ["oxides", "sulfides", "simple sulfides", "silicates", "tectosilicates", "feldspathoid group", "anhydrous sulfates", "sulfates", "sorosilicates", "nesosilicates", "feldspar group", "carbonates", "cyclosilicates", "pyroxene group", "inosilicates", "phyllosilicates", "semi-metallic elements", "native elements", "arsenates", "silica group", "garnet group", "apatite group", "true phosphates", "phosphates", "inosilicates", "amphibole group", "arsenates", "simple oxides", "zeolite group", "aragonite group", "anhydrous borates", "borates", "halides", "sulfosalts", "hydrous sulfates", "tourmaline group", "hydrous borates", "hydroxides", "mica group", "tellurides", "metallic elements", "chlorite group", "multiple oxides", "chromates", "humite group", "calcite group", "non-metallic elements", "tungstates and molybdates", "arsenides", "nitrates", "vanadates"]
ROCK_TYPES = ["metamorphic", "igneous", "sedimentary", "none"]


class Mineral:
  def __init__(self, name, url):

    # generate attribute dict
    self.name = name
    self.attributes = dict()
    self.image_urls = []
    self.image_paths = []
    self.scrape_text(url)
    if not os.path.exists('./' + self.name):
      os.mkdir('./' + self.name)

  def scrape_text(self, url):    
    # get text
    page = requests.get(url)
    html = etree.HTML(page.content)

    # hardness (average hardness values)
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblHardness"]')[0].text
    hardness = attr.split("-")
    hardness = sum(list(map(float, hardness)))/len(hardness)
    self.attributes['hardness'] = hardness

    # color
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblColor"]')[0].text
    color = attr.lower().replace(',', '').replace('.', '').replace('-', ' ').split(' ')
    self.attributes['color'] = list(set(color) & set(COLORS))

    # streak
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblStreak"]')[0].text
    streak = attr.lower().replace(',', '').replace('.', '').replace('-', ' ').split(' ')
    self.attributes['streak'] = list(set(streak) & set(STREAKS))

    # crystal system
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblCrystalSystem"]')[0].find('.//a').text
    crystal_system = attr.lower()
    if crystal_system not in CRYSTAL_SYSTEM:
      print("cystal system")
      print(crystal_system)
    self.attributes['crystal_system'] = crystal_system

    # specific gravity (average of values)
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblSpecificGravity"]')[0].text
    specific_gravity = attr.split("-")
    specific_gravity = sum(list(map(float, specific_gravity)))/len(specific_gravity)
    self.attributes['specific_gravity'] = specific_gravity

    # luster (no luster for now)
    #print(html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblLuster"]')[0].find('.//a'))
    #attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblLuster"]')[0].find('.//a').text
    #luster = attr.lower().split(' ')
    #print(list(set(luster) - set(LUSTERS)))
    #self.attributes['luster'] = list(set(luster) & set(LUSTERS))
 
    # fracture 
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblFracture"]')[0].findall('.//a')
    fracture = [a.text.lower() for a in attr]
    if len(list(set(fracture) - set(FRACTURES))) > 0:
      print("fracture")
      print(list(set(fracture) - set(FRACTURES)))
    self.attributes['fracture'] = list(set(fracture) & set(FRACTURES))
    
    # tenacity 
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblTenacity"]')[0].findall('.//a')
    tenacity = [a.text.lower() for a in attr]
    if len(list(set(tenacity) - set(TENACITYS))) > 0:
      print("tenacity")
      print(list(set(tenacity) - set(TENACITYS)))
    self.attributes['tenacity'] = list(set(tenacity) & set(TENACITYS))
    
    # groups
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblInGroup"]')[0].findall('.//a')
    groups = [a.text.lower().replace(';','') for a in attr]
    if len(list(set(groups) - set(GROUPS))) > 0:
      print("groups")
      print(list(set(groups) - set(GROUPS)))
    self.attributes['groups'] =  list(set(groups) & set(GROUPS))

    # rock type
    if len(html.xpath('.//tr[@id="ctl00_ContentPlaceHolder1_trRockType"]')) == 0:
      self.attributes['rock_type'] = ['none']
    else:
      attr = html.xpath('.//tr[@id="ctl00_ContentPlaceHolder1_trRockType"]')[0].xpath('.//a')[1:]
      rock_type = [a.text.lower().replace(',','') for a in attr]
      self.attributes['rock_type'] = list(set(rock_type) & set(ROCK_TYPES))

  def print_all_attributes(self):
    print(self.name)
    for att in self.attributes:
      print(att + ": " + str(self.attributes[att]))

  def print_attribute(self, att):
    print(att + ": " + self.attributes[att])

  def create_tf_record(self):
    base_dir = './tfrecords/'
    record_filename = base_dir + self.name + '.tfrecord'
    writer = tf.python_io.TFRecordWriter(record_filename)
    crystal_system_vec = self.crystal_system_to_vec().tostring()
    fracture_vec = self.fracture_to_vec().tostring()
    tenacity_vec = self.tenacity_to_vec().tostring()
    groups_vec = self.groups_to_vec().tostring()
    rock_type_vec = self.rock_type_to_vec().tostring()
    for image_name in self.image_paths:
      image = self.load_image(image_name).tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
        'crystal_system': _bytes_feature(crystal_system_vec),
        'fracture':       _bytes_feature(fracture_vec),
        'groups':         _bytes_feature(groups_vec),
        'rock_type':      _bytes_feature(rock_type_vec),
        'image':          _bytes_feature(image)}))
      writer.write(example.SerializeToString())
    
  def load_image(self, image_name):
    image = cv2.imread(image_name)
    return image 

  def crystal_system_to_vec(self):
    vec = np.zeros(len(CRYSTAL_SYSTEM))
    for i in xrange(len(CRYSTAL_SYSTEM)):
      if CRYSTAL_SYSTEM[i] in self.attributes['crystal_system']:
        vec[i] = 1.0
    vec = np.uint8(vec)
    return vec

  def fracture_to_vec(self):
    vec = np.zeros(len(FRACTURES))
    for i in xrange(len(FRACTURES)):
      if FRACTURES[i] in self.attributes['fracture']:
        vec[i] = 1.0
    vec = np.uint8(vec)
    return vec

  def tenacity_to_vec(self):
    vec = np.zeros(len(TENACITYS))
    for i in xrange(len(TENACITYS)):
      if TENACITYS[i] in self.attributes['tenacity']:
        vec[i] = 1.0
    vec = np.uint8(vec)
    return vec

  def groups_to_vec(self):
    vec = np.zeros(len(GROUPS))
    for i in xrange(len(GROUPS)):
      if GROUPS[i] in self.attributes['groups']:
        vec[i] = 1.0
    vec = np.uint8(vec)
    return vec

  def rock_type_to_vec(self):
    vec = np.zeros(len(ROCK_TYPES))
    for i in xrange(len(ROCK_TYPES)):
      if ROCK_TYPES[i] in self.attributes['rock_type']:
        vec[i] = 1.0
    vec = np.uint8(vec)
    return vec

  def add_image(self, url):
    image_name = './data/' + self.name + '/' + url.split('/')[-1]
    self.image_paths.append(image_name)
    self.image_urls.append(url)
    if not os.path.exists(image_name):
      img_data = requests.get(url).content
      with open(image_name, 'wb') as handler:
        handler.write(img_data)
    # change freakin file ext
    """file_ext = imghdr.what(image_name)
    os.remove(image_name)
    if file_ext is not None:
      image_name = image_name[:-3] + file_ext
      self.image_paths.append(image_name)
      with open(image_name, 'wb') as handler:
        handler.write(img_data)
    else:
      print(url)"""

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#def save_mineral(self, path='./'):

#Mineral('http://www.minerals.net/mineral/diopside.aspx')
