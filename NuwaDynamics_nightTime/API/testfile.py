from dataloader_taxibj import load_data
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
a,_,_,_,X=load_data(16,16,'/home/ansingh/NuwaDynamics/data/TaxiBJ',1)
b=next(iter(a))
# print(b)
print(b[0].size())
ans=b[0][0]
# plt.plot(ans)
tt=T.ToPILImage()
ans=tt(ans)
ans.show()
ans.save('lmao.png')
# print(b[0][0][0]-b[0][0][1])
# print(X.shape,Y.shape)
import h5py

data_path = './data/TaxiBJ/BJ13_M32x32_T30_InOut.h5'

# with h5py.File(data_path, 'r') as f:
#     print("Keys in the HDF5 file:", list(f.keys()))


