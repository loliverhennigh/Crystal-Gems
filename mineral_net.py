

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

    if not self.save_file_exits():
      self.generate_url_lists()
      self.save_to_pickle()
    else:
      self.load_from_pickle()

    self.mineral_list = dict()
    for mineral in self.mineral_urls:
      self.mineral_list[mineral] = Mineral(self.mineral_urls[mineral])

    for mineral in self.mineral_list:
      mineral.print_all_attributes()

  def generate_url_lists(self):
    # find all mineral urls i need
    mineral_urls = []
    base_url = 'http://www.minerals.net/MineralImages'
    abc_url_list = []
    #for l in ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','T','U','V','W','X','Y','Z']:
    for l in ['Z']:
      abc_url_list.append(base_url + '/' + l + '.aspx')
    for url in abc_url_list:
      print("scanning page " + url + " ...")
      page = requests.get(url)
      html = etree.HTML(page.content)
      base_1 = html.findall(".//div[@id]")
      for base in base_1:
        base_2 = base.findall(".//div") 
        if len(base_2) > 0:
          base_2 = base_2[0]
          for base in base_2.findall(".//a"):
            mineral_urls.append(base.attrib['href'])
      print("finished!")

    # get rid of repeats
    mineral_urls_store = []
    for i in mineral_urls:
      if i not in mineral_urls_store:
        mineral_urls_store.append(i)
    mineral_urls = mineral_urls_store

    # get proper list of mineral urls
    proper_mineral_urls = dict()
    for i in mineral_urls:
      if "Mineral" in i:
        i_type = i.split('/')[-1][:-5]
        proper_mineral_urls[i_type] = "http://www.minerals.net" + i

    # get proper list of image urls
    image_urls = defaultdict(list)
    image_count = 0
    print("generating list of images")
    for i in tqdm(mineral_urls):
      if "Image" in i:
        i_type = i.split('/')[-1][:-5]
        image_page = "http://www.minerals.net" + i[2:]
        print(image_page)
        html_page = requests.get(image_page).content
        image_index = html_page.find(".jpg&")
        html_page = html_page[image_index-200:image_index+100]
        html_page = html_page.split('"')
        image_url = "error"
        for url in html_page:
          if "jpg" in url:
            image_url = url
        image_urls[i_type].append("http://www.minerals.net/" + image_url)
        image_count += 1
 
    self.image_urls = image_urls   
    self.mineral_urls = proper_mineral_urls   
    print(image_urls)
    print(proper_mineral_urls)
    print(len(image_urls))
    print(len(proper_mineral_urls))
    print(image_count)
    

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


