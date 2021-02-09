import numpy as np

class ImplicitData:

    def __init__(self, user_list: list, item_list: list):
        self.userlist = np.array(user_list)
        self.itemlist = np.array(item_list)
        self.size = len(self.userlist)
        self.userset, self.userindices = np.unique(self.userlist, return_inverse=True)
        self.itemset, self.itemindices = np.unique(self.itemlist, return_inverse=True)
        self.maxuserid = len(self.userset) - 1
        self.maxitemid = len(self.itemset) - 1
        self.BuildMaps()

    def BuildMaps(self):
        self.useritems = []
        self.itemusers = []
        for u in range(self.maxuserid + 1):
            self.useritems.append([])
        for i in range(self.maxitemid + 1):
            self.itemusers.append([])
        for r in range(self.size):
            self.useritems[self.userindices[r]].append(self.itemindices[r])
            self.itemusers[self.itemindices[r]].append(self.userindices[r])

    def GetUserItems(self, user_id, internal = True):
        if internal:
            return self.useritems[user_id]
        uid = self.GetUserInternalId(user_id)
        return self.itemset[self.useritems[uid]]

    def GetItemUsers(self, item_id, internal = True):
        if internal:
            return self.itemusers[item_id]
        iid = self.GetItemInternalId(item_id)
        return self.userset[self.itemusers[iid]]

    def AddFeedback(self, user, item):
        self.size = self.size + 1
        self.userlist = np.append(self.userlist, user)
        self.itemlist = np.append(self.itemlist, item)
        if user not in self.userset:
            self.userset = np.append(self.userset, user)
            self.maxuserid = self.maxuserid + 1
            self.useritems.append([])
        user_id, = np.where(self.userset == user)[0]
        self.userindices = np.append(self.userindices, user_id)
        if item not in self.itemset:
            self.itemset = np.append(self.itemset, item)
            self.maxitemid = self.maxitemid + 1
            self.itemusers.append([])
        item_id, = np.where(self.itemset == item)[0]
        self.itemindices = np.append(self.itemindices, item_id)
        self.useritems[user_id].append(item_id)
        self.itemusers[item_id].append(user_id)
        return user_id, item_id

    def GetTuple(self, idx: int, internal: bool = False):
        if internal:
            return self.userindices[idx], self.itemindices[idx]
        return self.userlist[idx], self.itemlist[idx]

    def GetUserInternalId(self, user):
        user_id, = np.where(self.userset == user)
        if len(user_id):
            return user_id[0]
        return -1

    def GetItemInternalId(self, item):
        item_id, = np.where(self.itemset == item)
        if len(item_id):
            return item_id[0]
        return -1

    def GetUserExternalId(self, user_id:int):
        if user_id > -1 and item_id <= self.maxuserid:
            return self.userset[user_id]
        return ""

    def GetItemExternalId(self, item_id:int):
        if item_id > -1 and item_id <= self.maxitemid:
            return self.itemset[item_id]
        return ""
