import cv2
import os
import pandas as pd
from shutil import copyfile

s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s



bill_code_path = 'transformed/bill_code'
market_name_path = 'transformed/market_name'
date_path = 'transformed/date'
file_name = 'Template_OCR BILL COOP 2_251220.xlsx'
file_name_25 = 'ORC_TEMPLATE01_total.xlsx'
path_grouth_truth = 'transformed/images'

df = pd.read_excel(file_name, sheet_name='FULL',skiprows=1)
df = df.iloc[:,1:5]
df.dropna(inplace=True)

df_25 = pd.read_excel(file_name_25, sheet_name='Template', skiprows=1,dtype=str)
df_25 = df_25.iloc[:,1:5]
df_25["Date"] = pd.to_datetime(df_25["Date"])
df_25.dropna(inplace=True)

imagesName = []
imagesName_25 = []
for image in os.listdir(market_name_path):
    condition_1 = 'txt' not in image and '_25_12' not in image[:-11]
    condition_2 = 'txt' not in image and '_25_12' in image[:-11]
    if condition_1:
        imagesName.append(image)
    if condition_2:
        imagesName_25.append(image)
print(imagesName)
print(imagesName_25)
i = 1
t = 1
name_old = df.iloc[0,0]
for index, row in df.iterrows():
    market_name = row['Market Name']
    image_name = f'{str(row["Image Name"]).lower()}_padded.jpg'
    date = f'Ngay: {row["Date"].day}/{row["Date"].month}/{row["Date"].year}'
    bill_code = f'So HD: {row["Bill Code"]}'
    if image_name != name_old:
        if image_name in imagesName:
            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_market_name_{i}.gt.txt'),'w+') as f:
                src = os.path.join(market_name_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_market_name_{i}.jpg')
                try:
                    # do something
                    copyfile(src, dst)
                    market_name = remove_accents(market_name)
                    f.write(market_name)
                except FileNotFoundError:
                    # handle ValueError exception
                    pass
            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_bill_code_{i}.gt.txt'),'w+') as f:
                src = os.path.join(bill_code_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_bill_code_{i}.jpg')
                try:
                    # do something
                    copyfile(src, dst)
                    f.write(bill_code)
                except FileNotFoundError:
                    # handle ValueError exception
                    pass

            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_date_{i}.gt.txt'),'w+') as f:
                src = os.path.join(date_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_date_{i}.jpg')
                try:
                    # do something
                    copyfile(src, dst)
                    f.write(date)
                except FileNotFoundError:
                    # handle ValueError exception
                    pass

            i+=1
            name_old = image_name
        

name_old = df_25.iloc[0,0]
for index, row in df_25.iterrows():
    market_name = row['Market Name']
    image_name = f'{str(row["Image Name"]).lower()}_25_12_padded.jpg'
    date = f'Ngay: {row["Date"].month}/{row["Date"].day}/{row["Date"].year}'
    bill_code = f'So HD: {row["Bill Code"]}'
    if image_name != name_old:
        if image_name in imagesName_25:
            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_market_name_{i}.gt.txt'),'w+') as f:
                src = os.path.join(market_name_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_market_name_{i}.jpg')
                
                copyfile(src, dst)
                market_name = remove_accents(market_name)
                f.write(market_name)
            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_bill_code_{i}.gt.txt'),'w+') as f:
                src = os.path.join(bill_code_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_bill_code_{i}.jpg')
                copyfile(src, dst)
                f.write(bill_code)
            with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}_date_{i}.gt.txt'),'w+') as f:
                src = os.path.join(date_path,image_name)
                dst = os.path.join(path_grouth_truth,f'{image_name[:-4]}_date_{i}.jpg')
                copyfile(src, dst)
                f.write(date)
            i+=1
            name_old = image_name

# for index, row in df.iterrows():
#     market_name = row['Market Name']
#     image_name = f'{str(row["Image Name"]).lower()}_padded.jpg'
#     date = f'Ngay: {row["Date"].day}/{row["Date"].month}/{row["Date"].year}'
#     bill_code = f'So HD: {row["Bill Code"]}'
#     if image_name in imagesName:
#         with open(os.path.join(path_grouth_truth,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             market_name = remove_accents(market_name)
#             f.write(market_name)
#         with open(os.path.join(bill_code_path,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             f.write(bill_code)
#         with open(os.path.join(date_path,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             f.write(date)

# for index, row in df_25.iterrows():
#     market_name = row['Market Name']
#     image_name = f'{str(row["Image Name"]).lower()}_25_12_padded.jpg'
#     date = f'Ngay: {row["Date"].day}/{row["Date"].month}/{row["Date"].year}'
#     bill_code = f'So HD: {row["Bill Code"]}'
#     if image_name in imagesName_25:
#         with open(os.path.join(market_name_path,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             market_name = remove_accents(market_name)
#             f.write(market_name)
#         with open(os.path.join(bill_code_path,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             f.write(bill_code)
#         with open(os.path.join(date_path,f'{image_name[:-4]}.gt.txt'),'w+') as f:
#             f.write(date)

