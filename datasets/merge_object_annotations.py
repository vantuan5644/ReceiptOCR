import os
import numpy as np
import xml.etree.ElementTree as ET



def get_file_name(path):
    base_dir = os.path.dirname(path)
    file_name, ext = os.path.splitext(os.path.basename(path))
    ext = ext.replace(".", "")
    return base_dir, file_name, ext


def get_modified_xml(src_xml, dst_dir):
    xmlRoot = ET.parse(src_xml).getroot()

    last_member = None
    anchor = {'xmin': 100000, 'ymin': 100000, 'xmax': 0, 'ymax': 0}

    for member in xmlRoot.findall('object'):
        if member.find('name').text == 'product_attributes':
            last_member = member
            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            anchor['xmin'] = min(anchor['xmin'], xmin)
            anchor['ymin'] = min(anchor['ymin'], ymin)
            anchor['xmax'] = max(anchor['xmax'], xmax)
            anchor['ymax'] = max(anchor['ymax'], ymax)
            xmlRoot.remove(member)

    if last_member is not None:
        bndbox = last_member.find('bndbox')

        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        xmin.text = str(anchor['xmin'])
        ymin.text = str(anchor['ymin'])
        xmax.text = str(anchor['xmax'])
        ymax.text = str(anchor['ymax'])

        xmlRoot.append(last_member)

        tree = ET.ElementTree(xmlRoot)
        # tree.write('{}/{}.xml'.format(output_path, file_name, ext))
        tree.write(os.path.join(dst_dir, os.path.split(src_xml)[1]))

src_dir = './COOP/transformed'
dst_dir = './COOP/transformed_'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
for file in os.listdir(src_dir):
    if file.lower()[-4:] == '.xml':
        file = os.path.join(src_dir, file)
        get_modified_xml(src_xml=file, dst_dir=dst_dir)