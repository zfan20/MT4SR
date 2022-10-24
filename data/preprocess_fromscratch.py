import gzip
from collections import defaultdict
from datetime import datetime
import os
import copy
import json
import numpy as np

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
      yield eval(l)


DATASET = 'Office_Products'
dataname = '/home/zfan/BDSC/projects/datasets/reviews_' + DATASET + '_5.json.gz'
meta_path = '/home/zfan/BDSC/projects/datasets/newmetadata/meta_{}.json.gz'.format(DATASET)
if not os.path.isdir('./'+DATASET):
    os.mkdir('./'+DATASET)
train_file = './'+DATASET+'/train.txt'
valid_file = './'+DATASET+'/valid.txt'
test_file = './'+DATASET+'/test.txt'


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    countU[rev] += 1
    countP[asin] += 1

usermap = dict()
usernum = 0
itemmap = dict()
itemnum = 0
User = dict()


for one_interaction in parse(dataname):
    rev = one_interaction['reviewerID']
    asin = one_interaction['asin']
    time = float(one_interaction['unixReviewTime'])
    if countU[rev] < 5:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
        usernum += 1
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemid = itemnum
        itemmap[asin] = itemid
        itemnum += 1
    User[userid].append([itemid, time])

for userid in User.keys():
    User[userid].sort(key=lambda x: x[1])

Item = {}
relationships_ind_map = {}
r_ind = 0
with gzip.open(meta_path, 'r') as f:
    for line in f:
        e = eval(line)
        if e['asin'] in itemmap:
            related_infor = {}
            for rel, rel_list in e.get('related', {}).items():
                if rel not in relationships_ind_map:
                    relationships_ind_map[rel] = r_ind
                    r_ind += 1
                new_rel_list = []
                for eachitem in rel_list:
                    if eachitem in itemmap:
                        new_rel_list.append(itemmap[eachitem])
                related_infor[rel] = new_rel_list
            Item[itemmap[e['asin']]] = {'related': related_infor}


user_train = {}
user_validation = {}
user_testing = {}
for user in User:
    nfeedback = len(User[user])
    if nfeedback < 3:
        user_train[user] = [itemid for itemid, time in User[user]]
        user_validation[user] = []
        user_testing[user] = []
    else:
        user_train[user] = [itemid for itemid, time in User[user][:-2]]
        user_validation[user] = []
        user_validation[user].append(User[user][-2][0])
        user_testing[user] = []
        user_testing[user].append(User[user][-1][0])





#Item_relationship_mask_mat_completeseqs, relationships_ind
print(relationships_ind_map)
Item_relationship_mask_mat_completeseqs = {}
for user in user_train.keys():
    completeseq = user_train[user] + user_validation[user] + user_testing[user]
    if len(completeseq) > 200:
        completeseq = completeseq[-200:]

    mask = np.zeros((len(relationships_ind_map), len(completeseq), len(completeseq)), dtype=np.int32)

    for ind in range(len(completeseq)):
        for next_ind in range(len(completeseq)):
            related_infor = Item[completeseq[ind]]['related']
            next_related_infor = Item[completeseq[next_ind]]['related']
            for eachrel, rel_items in related_infor.items():
                if completeseq[next_ind] in rel_items:
                    mask[relationships_ind_map[eachrel], ind, next_ind] = 1
            for eachrel, rel_items in next_related_infor.items():
                if completeseq[ind] in rel_items:
                    mask[relationships_ind_map[eachrel], ind, next_ind] = 1

    Item_relationship_mask_mat_completeseqs[user] = mask


#user_valid = defaultdict(list)
#user_test = defaultdict(list)
#for u, ituple in user_validation.items():
#    if len(ituple) > 0:
#        user_valid[u] = [ituple[1]]
#for u, ituple in user_testing.items():
#    if len(ituple) > 0:
#        user_test[u] = [ituple[1]]
new_dataset = [user_train, user_validation, user_testing, Item, Item_relationship_mask_mat_completeseqs, relationships_ind_map, usernum, itemnum]
np.save('./'+DATASET+'Partitioned_5core', new_dataset)


print(usernum, itemnum)

def writetofile(data, dfile):
    with open(dfile, 'w') as f:
        for u, ilist in sorted(data.items()):
            for i in ilist:
                f.write(str(u) + '\t'+ str(i) + "\n")

#writetofile(user_train, train_file)
#writetofile(user_valid, valid_file)
#writetofile(user_test, test_file)


num_instances = sum([len(ilist) for _, ilist in user_train.items()])
num_instances += sum([len(ilist) for _, ilist in user_validation.items()])
num_instances += sum([len(ilist) for _, ilist in user_testing.items()])
print('total user: ', usernum)
print('total instances: ', num_instances)
print('total items: ', itemnum)
print('density: ', num_instances / (usernum * itemnum))
print('valid #users: ', len(user_validation))
numvalid_instances = sum([len(ilist) for _, ilist in user_validation.items()])
print('valid instances: ', numvalid_instances)
numtest_instances = sum([len(ilist) for _, ilist in user_testing.items()])
print('test #users: ', len(user_testing))
print('test instances: ', numtest_instances)
avg_rel_density = 0.
for u, mask in Item_relationship_mask_mat_completeseqs.items():
    mask_shape = mask.shape
    nonzeros = np.sum(mask)
    avg_rel_density += nonzeros / np.prod(mask_shape)

print('avg rel density: ', avg_rel_density / len(Item_relationship_mask_mat_completeseqs))

#Item relationships
num_pairs = 0
for eachitem, related_item in Item.items():
    related_dict = related_item['related']
    for rel, rel_list in related_dict.items():
        num_pairs += len(rel_list)

print('total num item pairs: ', num_pairs)
print('avg num item pairs/item: ', num_pairs/len(Item))
