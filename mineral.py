
from io import StringIO, BytesIO
import requests
from lxml import etree
import os

COLORS = ["gray", "black", "white", "green", "yellow", "blue", "pink", "orange", "red", "colorless", "beige", "brown", "purple", "multicolored", "steel", "silver", "pale", "cream", "tin"]
STREAKS = ["gray", "black", "white", "green", "yellow", "blue", "pink", "orange", "red", "colorless", "beige", "brown", "purple", "multicolored", "steel", "silver", "pale", "cream", "tin"]
LUSTERS = ["adamantine", "resinous"]
FRACTURES = ["subconchoidal", "splintery", "conchoidal", "uneven"]
TENACITYS = ["brittle", "malleable", "sectile", "fibrous", "elastic"]
GROUPS = ["oxides", "sulfides", "simple sulfides", "silicates", "tectosilicates", "feldspathoid group", "anhydrous sulfates", "sulfates", "sorosilicates", "nesosilicates", "feldspar group", "carbonates", "cyclosilicates", "pyroxene group", "inosilicates", "phyllosilicates", "semi-metallic elements", "native elements", "arsenates", "silica group", "garnet group", "apatite group", "true phosphates", "phosphates", "inosilicates", "amphibole group", "arsenates", "simple oxides", "zeolite group", "aragonite group", "anhydrous borates", "borates", "halides", "sulfosalts", "hydrous sulfates", "tourmaline group", "hydrous borates", "hydroxides", "mica group"]
ROCK_TYPES = ["metamorphic", "igneous", "sedimentary", "none"]


class Mineral:
  def __init__(self, name, url):

    # generate attribute dict
    print(name)
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
    self.attributes['fracture'] = list(set(fracture) & set(FRACTURES))
    
    # tenacity 
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblTenacity"]')[0].findall('.//a')
    tenacity = [a.text.lower() for a in attr]
    self.attributes['tenacity'] = list(set(tenacity) & set(TENACITYS))
    
    # groups
    attr = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblInGroup"]')[0].findall('.//a')
    groups = [a.text.lower().replace(';','') for a in attr]
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

  def add_image(self, url):
    self.image_urls.append(url)
    img_data = requests.get(url).content
    image_name = './' + self.name + '/' + url.split('/')[-1]
    if not is_jpg(img_data):
      image_name = image_name[:-4] + '.png'
    self.image_paths.append(image_name)
    with open(image_name, 'wb') as handler:
      handler.write(img_data)

def is_jpg(data):
    if data[:4] != '\xff\xd8\xff\xe0': return False
    if data[6:] != 'JFIF\0': return False
    return True

#def add_image(self, url):

#def save_mineral(self, path='./'):

#Mineral('http://www.minerals.net/mineral/diopside.aspx')
