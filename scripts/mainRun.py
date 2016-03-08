#!/usr/bin/env python
import os
'''
inte än bestämt om vi har separata funktionsfiler eller allt i ett script. 
'''

if len(sys.argv) == 4:
    source_dir  = sys.argv[1]
    haar_model  = sys.argv[2]
    caffe_model = sys.argv[3]
    deploy_file  = sys.argv[4]
else:
    print("usage: %s source_dir haar_model caffe_model caffe_deploy_file" % __file__)
    sys.exit(1)

for image in os.listdir(source_dir):
    source_paths.append='%s/%s'%(source_dir, image)

create_crops(haar_model, source_paths, crop_path)

classifications = classify(caffe_model, deploy_file, crop_path)


