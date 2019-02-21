from config import *
data_dir = 'data/'

# train data
# tra_loc = np.zeros([2, num_user], dtype = float)
# tra_loc[0] = np.random.uniform(70, 90, [num_user])
# tra_loc[1] = np.random.uniform(70, 90, [num_user])
# np.savetxt(data_dir + 'tra_user%s_loc88' %  str(num_user), tra_loc)

# test data
tes_loc = np.random.uniform(0, 20, [500, 100])
np.savetxt(data_dir + 'tes_loc00_20', tes_loc)

#  test = np.loadtxt(data_dir + 'tes_loc80_00')
# print(test[0,:state_dim])
# print(test[0,state_dim:2*state_dim])


































































































