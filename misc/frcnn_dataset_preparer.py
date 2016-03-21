#!/usr/bin/env python

import os
import argparse
import xml.etree.ElementTree as ET
import shutil

def add_object_xml(name, bndbox, annotation):
    _object = ET.SubElement(annotation, 'object')
    _object_name = ET.SubElement(_object, 'name')
    _object_name.text = name

    _object_bndbox = ET.SubElement(_object, 'bndbox')
    xmin = ET.SubElement(_object_bndbox, 'xmin')
    ymin = ET.SubElement(_object_bndbox, 'ymin')
    xmax = ET.SubElement(_object_bndbox, 'xmax')
    ymax = ET.SubElement(_object_bndbox, 'ymax')

    xmin.text = '%s' % bndbox[0]
    ymin.text = '%s' % bndbox[1]
    xmax.text = '%s' % bndbox[2]
    ymax.text = '%s' % bndbox[3]

    return annotation


def create_xml(folder_name, image_name, database, object_name, bndbox):
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_name

    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    source = ET.SubElement(annotation, 'source')

    database_source = ET.SubElement(source, 'database')
    database_source.text = database

    annotation_source = ET.SubElement(source, 'annotation')
    annotation_source.text = 'Traffic2016'

    image_source = ET.SubElement(source, 'image')
    image_source.text = ''

    _size = ET.SubElement(annotation, 'size')
    size_width = ET.SubElement(_size, 'width')
    size_height = ET.SubElement(_size, 'height')
    size_depth = ET.SubElement(_size, 'depth')
    size_width.text = '%s' % size[0]
    size_width.text = '%s' % size[1]
    size_depth.text = '%s' % size[2]

    return add_object_xml(object_name, bndbox, annotation)

if __name__ == "__main__":
    ap = argparse.ArgumentParser
    ap.add_argument("-t", "--txt", required=True,
                    help="Path to the annotation txt")
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to the output directory")
    ap.add_argument("-i", "--imdir", required=True,
                    help="Path to the image directory")
    args = vars(ap.parse_args)
    txt_dir = args["txt"]
    target_dir = args["outdir"]
    image_dir = args["imdir"]

    if not os.path.exists(target_dir):
        os.makedirs('%s/data/Images'%target_dir)
        os.makedirs('%s/data/Annotations'%target_dir)
        os.makedirs('%s/data/ImageSets'%target_dir)
    with open(txt_dir) as read_f:
        database = 'Generic database name' # Can be changed to desired name of the database
        folder_name = imdir.strip("/")
        folder_name = folder_name[-2]

        last_image = None
        for line in read_f:
            line = line.strip("\n")
            line = line.strip(";")
            image_name = line[0]
            object_name = line[5]
            bndbox[0:4] = line[1:5] 
            line_dir = '%s/%s.xml' % (target_dir, image_name[0].strip("."))

            if not os.path.exists(line_dir):
                annotation = create_xml(
                    folder_name, image_name, database, object_name, bndbox)
            else:
                annotation = ET.parse(line_dir)
                annotation = add_object_xml(object_name, bndbox, annotation)

            annotation.write('%s.xml' % line_dir)

            with open('%s/data/ImageSets/train.txt'%target_dir, 'w') as write_f:
                if image_name != last_image:
                    write_f.write('%s\n'%image_name[0].strip("."))
            last_image = image_name
            
            if not os.path.exists('%s/data/Images/%s'%(target_dir,image_name)):
                shutil.copy('%s/%s'%(image_dir,image_name),('%s/data/Images'%target_dir))


