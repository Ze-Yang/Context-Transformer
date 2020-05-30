from data.voc0712 import VOC_CLASSES
from data.config import VOCroot
import sys
import os
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

image_sets = [('2007', 'trainval'), ('2012', 'trainval')]
annopath = os.path.join('%s', 'Annotations', '%s.xml')

for (year, name) in image_sets:
    rootpath = os.path.join(VOCroot, 'VOC' + year)
    ids = list()
    for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        ids.append((rootpath, line.strip()))
    for split in range(1, 4):
        base_img = list()
        for img_id in ids:
            anno = ET.parse(annopath % img_id).getroot()
            base_only = True
            for obj in anno.iter('object'):
                name = obj.find('name').text.lower().strip()
                class_to_ind = dict(
                    zip(VOC_CLASSES[split], range(len(VOC_CLASSES[split]))))
                label_idx = class_to_ind[name]
                if label_idx > 15:
                    base_only = False
                    break
            if base_only:
                base_img.append(str(img_id[1]))
        print('Number of base images from VOC {} for split {}: {}'.format(year, split, len(base_img)))
        save_file = os.path.join(rootpath, 'ImageSets', 'Main', 'trainval_split{}.txt'.format(split))
        print('Saving to {}'.format(save_file))
        with open(save_file, 'w') as f:
            f.write('\n'.join(base_img) + '\n')

