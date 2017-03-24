#!/usr/bin/env python

import os
import argparse
# import xml.etree.ElementTree as ET
import lxml.etree as ET
import shutil
import csv


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


def create_xml(
    folder_name,
    image_name,
    database,
    object_name,
    bndbox,
     img_size):
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = folder_name

    filename = ET.SubElement(annotation, 'filename')
    filename.text = image_name

    source = ET.SubElement(annotation, 'source')
    database_source = ET.SubElement(source, 'database')
    database_source.text = database

    annotation_source = ET.SubElement(source, 'annotation')
    annotation_source.text = 'GTSRB2016'

    image_source = ET.SubElement(source, 'image')
    image_source.text = ''

    _size = ET.SubElement(annotation, 'size')
    size_width = ET.SubElement(_size, 'width')
    size_height = ET.SubElement(_size, 'height')
    size_depth = ET.SubElement(_size, 'depth')
    size_width.text = str(img_size[0])
    size_height.text = str(img_size[1])
    size_depth.text = str(img_size[2])

    return add_object_xml(object_name, bndbox, annotation)


def bndbox_trans_check(bndbox, img_size):

    bndbox = [int(s) for s in bndbox]
    if bndbox[0] < 0:
        bndbox[0] = 0
    if bndbox[1] < 0:
        bndbox[1] = 0
    if bndbox[2] > img_size[0]:
        bndbox[2] = img_size[0]
    if bndbox[3] > img_size[1]:
        bndbox[3] = img_size[1]

    return bndbox


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", required=True,
                    help="Path to the output directory")
    ap.add_argument("-i", "--indir", required=True,
                    help="Path to the root image directory")
    args = vars(ap.parse_args())
    target_dir = args["outdir"]
    source_dir = args["indir"]

    if not os.path.exists(target_dir):
        os.makedirs('%s/data/Images' % target_dir)
        os.makedirs('%s/data/Annotations' % target_dir)
        os.makedirs('%s/data/ImageSets' % target_dir)
    image_list = []
    for class_name in os.listdir(source_dir):
        class_dir = os.path.join(source_dir, class_name)
        csvs = [os.path.join(class_dir, f)
                for f in os.listdir(class_dir) if f.endswith(".csv")]
        assert len(csvs) == 1
        gt_filename = csvs[0]
        with open(gt_filename) as gt_file:
            gt_reader = csv.reader(gt_file, delimiter=";")
            # Can be changed to desired name of the database
            database = 'gtsrb'
            folder_name = source_dir.split("/")[-1]
            last_image = None
            gt_header = gt_reader.next()
            for row in gt_reader:
                image_name = row[0]
                image_name_with_class = "%s_%s" % (class_name, row[0])
                object_name = 'sign'  # TODO fix annotation script + row[8]
                bndbox = row[3:7]
                annotation_xml = os.path.join(target_dir, "data",
                                                "Annotations",
                                                "%s.xml" % image_name_with_class.split(".")[0])
                img_size = [int(s) for s in row[1:3]]
                img_size.append(3)
                bndbox = bndbox_trans_check(bndbox, img_size)
                if not os.path.exists(annotation_xml):
                    annotation = create_xml(
                        folder_name, image_name_with_class, database,
                        object_name, bndbox, img_size)
                else:
                    tree = ET.parse(annotation_xml)
                    annotation = tree.getroot()
                    annotation = add_object_xml(
                        object_name, bndbox, annotation)
                try:
                    tree = ET.ElementTree(annotation)
                    tree.write(annotation_xml, pretty_print=True)
                except:
                    import pdb
                    pdb.set_trace()  # XXX BREAKPOINT
                    raise
                if image_name != last_image:
                    image_list.append('%s\n' %
                                    image_name_with_class.split(".")[0])
                last_image = image_name

                if not os.path.exists(
                        os.path.join(target_dir, "data", "Images",
                                        image_name_with_class)):
                    shutil.copy(
                        os.path.join(class_dir, image_name),
                        os.path.join(target_dir, "data", "Images",
                                        image_name_with_class))
    with open('%s/data/ImageSets/train.txt' % target_dir, 'w') as write_f:
        write_f.writelines(image_list)
