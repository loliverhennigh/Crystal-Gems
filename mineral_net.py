

from io import StringIO, BytesIO
import requests
from lxml import etree
from collections import defaultdict
import re
from tqdm import *
import pickle
import os.path
from mineral import *


class Mineral_Net:
  def __init__(self):

    # make store lists
    self.minerals = dict()
    self.mineral_names = []
    self.image_urls = defaultdict(list)
    self.mineral_urls = dict()
    print("scraping urls")
    self.generate_url_lists()

    print("creating list of minerals")
    for name in tqdm(self.mineral_names):
      self.minerals[name] = Mineral(name, self.mineral_urls[name])

    print("downloading images")
    for name in tqdm(self.mineral_names):
      for image in self.image_urls[name]:
        self.minerals[name].add_image(image)

    print("creating tf records")
    for name in tqdm(self.mineral_names):
      self.minerals[name].create_tf_record()
    #for name in self.mineral_names:
    #  self.minerals[name].print_all_attributes()

  def generate_url_lists(self):
    # find all mineral urls i need
    base_url = 'http://www.minerals.net/MineralImages'
    abc_url_list = []
    #for l in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z']:
    for l in ['N']:
      abc_url_list.append(base_url + '/' + l + '.aspx')
    for url in abc_url_list:
      print("scanning page " + url + " ...")
      page = requests.get(url)
      html = etree.HTML(page.content)
      images = html.xpath('.//a')
      for i in images:
        if 'href' in i.attrib.keys():
          if '../Image/' in i.attrib['href']:
            name = i.attrib['href'].split('/')[-1][:-5]
            image = i.find('.//img')
            self.image_urls[name].append(image.attrib['src'])

    # modify image urls to make larger (remove -t flag)
    for name in self.image_urls:
      for i in xrange(len(self.image_urls[name])):
        #self.image_urls[name][i] = 'http://www.minerals.net/' + self.image_urls[name][i][:-6] + ".jpg"
        self.image_urls[name][i] = 'http://www.minerals.net/' + self.image_urls[name][i]

    # make list of mineral names
    for name in self.image_urls:   
      self.mineral_names.append(name)

    # make list of mineral url pages
    for name in self.mineral_names:
      self.mineral_urls[name] = 'http://www.minerals.net/mineral/' + name + '.aspx'


  def save_file_exits(self):
    if os.path.isfile('./pickle_minerals.pickle'):
      return True
    else:
      return False

  def save_to_pickle(self):
    with open('./pickle_minerals.pickle', 'wb') as output:
      pickle.dump(self, output)

  def load_from_pickle(self):
    with open('./pickle_minerals.pickle', 'rb') as input:
      self.__dict__.update(pickle.loads('./pickle_minerals.pickle').__dict__)
    
 
Mineral_Net()


