import glob
#import pydicom
import imageio
import numpy as np
#from skimage.transform import rotate
#from ipywidgets import interact
#from sklearn.preprocessing import MinMaxScaler
import os
import winsound
from matplotlib import pyplot as plt
import nibabel
import matplotlib.animation as animate
#from skimage.util import montage 
from PIL import Image

def beep():
    frequency = 700  # Set Frequency To 2500 Hertz
    duration = 950  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
    
def join_paths(path1,path2):
    return(os.path.join(path1,path2))

def load_nifti(image_path):
    '''Loading a nifti image'''
    load_img = nibabel.load(image_path).get_fdata()
    return(load_img)

def get_slice(loaded,layer_num):
    '''Get a layer from a nifti loaded image'''
    return(loaded[:,:,layer_num])

def show_nifti(image_path,slice_num):
    '''Display nifti image (mri)'''
    loaded = load_nifti(image_path)
    slice_img = loaded[:,:,slice_num]
    plt.imshow(slice_img,cmap='gray')
    plt.show()

def animate_img(image_path):
    '''Loading a slide animation of the mri'''
    loaded = load_nifti(image_path)
    def visualize_3d(layer):
        plt.figure(figsize=(10, 5))
        plt.imshow(loaded[:, :, layer], cmap='gray')
        plt.axis('off')
        return layer
    interact(visualize_3d, layer=(0, loaded.shape[2] - 1))

def animate_labels(img_path_seg):
    '''Display the tumorous part (the segmented part) with a slide'''
    mask = load_nifti(img_path_seg)
    layer = 50
    classes_dict = {
        'Normal': 0.,
        'Edema': 1.,
        'Non-enhancing tumor': 2.,
        'Enhancing tumor': 3. 
    }

    def visualize_3d_labels(layer):
        mask = load_nifti(img_path_seg)
        plt.imshow(mask[:,:,layer],cmap='BuPu')
        plt.axis('off')
        plt.tight_layout()

    interact(visualize_3d_labels, layer=(0, mask.shape[2] - 1));

def show_all_layers(image_path):
    '''Display all layers of the mri'''
    loaded = load_nifti(image_path)
    fig, ax1 = plt.subplots(1, 1, figsize = (20,20))
    ax1.imshow(rotate(montage(loaded[:,:,:]), 90, resize=True), cmap ='gray')
    plt.axis('off')

# define function to create gif from image data
def create_gif(input_image, title='.gif', filename='test.gif',cmap='gray'):
    '''creates a gif from image data'''
    # see example from matplotlib documentation

    images = []
    input_image_data = input_image.get_fdata()    
    fig = plt.figure()    
    for i in range(len(input_image_data)):
        im = plt.imshow(input_image_data[i], animated=True,cmap=cmap)
        images.append([im])
    
    ani = animate.ArtistAnimation(fig, images, interval=25,\
        blit=True, repeat_delay=500)
    plt.axis('off')
    ani.save(filename)
    plt.show()

def df_row2list(row,beg=0,end=-1):
    '''convert a dataframe row to a list'''
    if end == -1:
        conv_list = row.values.flatten().tolist()[beg:]
        return(conv_list)
    conv_list = row.values.flatten().tolist()[beg:end:]
    return(conv_list)

def get_paths(abs_path,row_df):
    '''Joining basename and parent_directory'''
    image_paths = df_row2list(row_df,1,-2)
    print(image_paths)
    image_paths_abs = []
    for base_path in image_paths:
        image_paths_abs.append(os.path.join(abs_path,base_path))
    return(image_paths_abs)

def load_the_mris(list_of_paths,scan_type,slice,values_mris):
    '''Display the diffenrent scan types of one patient'''
    # Load NIfTI images
    images = [load_nifti(path) for path in list_of_paths]

    # Set the desired height and width
    fig_height = 20  # Adjust as needed
    fig_width = 20  # Adjust as needed

    # Create a figure with 1 row and 4 columns
    fig, axs = plt.subplots(1, 4, figsize=(fig_width, fig_height))

    # Show each image in a separate subplot
    for i in range(4):
        axs[i].imshow(images[i][:,:,slice], cmap='gray')
        axs[i].set_title(f"{scan_type[i]}",fontsize=16) 
        axs[i].axis('off')

    # Adjust spacing and display the figure
    plt.suptitle(f"ID: {values_mris[0]} | Index: {values_mris[1]}", fontsize=16, fontweight='bold',y=0.6,x=0.5)
    plt.show()

def save_nif2png_slices(dim,outputPath,scan_File_Path):
    
    # Load the scan and extract data using nibabel 
    scan = nibabel.load(scan_File_Path)
    scanArray = scan.get_fdata()        
    scanArrayShape = scanArray.shape
    
    # Generate the slices through the three axes     and store them in an array
    for z_slice_number in range(scanArrayShape[dim]):
        if dim == 2:
            slice = scanArray[:, :, z_slice_number]
        elif dim == 1:
            slice = scanArray[:, z_slice_number,:]
        else :
            slice = scanArray[z_slice_number,:,:]
        
        # Save the slice as .png image
        min_value = np.min(slice)
        max_value = np.max(slice)

        if min_value == max_value:
            normalized_array = slice - min_value
        else:
            normalized_array = (slice - min_value) / (max_value - min_value) * 65535
            
        # Save the array as a 16-bit grayscale PNG image
        if not (np.all(normalized_array == 0) ):
            uint16_array = normalized_array.astype(np.uint16)
            imageio.imwrite(outputPath+'\Slice_dim'+str(dim)+'_'+str(z_slice_number)+'.png', uint16_array)

def images_based_on_scan_type(dataframe,df_type):
    name_t1c = dataframe["t1c"].tolist()
    name_t1n = dataframe["t1n"].tolist()
    name_t2f = dataframe["t2f"].tolist()
    name_t2w = dataframe["t2w"].tolist()
    if df_type == 'train':
        name_seg = dataframe["seg"].tolist()
        return(name_seg,name_t1c,name_t1n,name_t2f,name_t2w)
    else:
        return('0',name_t1c,name_t1n,name_t2f,name_t2w)

def savin_dims(path_scan_type,str_scan_type,output_parent_dir_path,parent_dir,name_folders):
    for name,scan_type in zip(name_folders,path_scan_type):
        #Define the filepath to your NIfTI scan
        scan_path = os.path.join(parent_dir,scan_type)
        output_path = os.path.join(os.path.join(output_parent_dir_path,name),str_scan_type)
        #save_nif2png_slices(0,output_path,scan_path)
        #save_nif2png_slices(1,output_path,scan_path)
        save_nif2png_slices(2,output_path,scan_path)

def rm_useless_slices(scan_type,refs, full_slices):
    set_difference = set(full_slices) - set(refs)
    list_diff = sorted(list(set_difference))

    for useless in list_diff:
        temp = os.path.join(scan_type,useless)
        #!rm $temp
        print('removed!')

def metadata_rows(scan_type,paths,split):
    curr_row = []
    for path in paths:
        print(path)
        p_id = path.split('-')[-2]
        t_stamp = path.split('-')[-1]
        folder = os.path.join(path,scan_type)
        images_paths = os.listdir(folder)
        for image_name in images_paths:
            curr_row.append([image_name,p_id,t_stamp,scan_type,split])
        print([image_name,p_id,t_stamp,scan_type,split])
    return(curr_row)

    import numpy as np


def normalize_image(image):
    # Reshape the image to 2D array if it's multi-dimensional
    if len(image.shape) > 2:
        image = image.reshape(-1, image.shape[-1])

    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform the image
    normalized_image = scaler.fit_transform(image)

    # Reshape the normalized image back to its original shape
    normalized_image = normalized_image.reshape(image.shape)

    return normalized_image

def replace_pixels(image_array:np.array):
    # Define the pixel values to change
    original_values = [21845, 43690, 65535]
    new_values = [1, 2, 3]

    # Replace the original pixel values with the new values
    for orig_val, new_val in zip(original_values, new_values):
        image_array[image_array == orig_val] = new_val
    return image_array

def replace_non_zero_pixels(image_array):
    # Replace non-zero pixels with 1
    image_array[image_array != 0] = 1

    # Convert the NumPy array back to a PIL image
    modified_image = Image.fromarray(image_array)

    return modified_image

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data

def visualize_sample(
    brats21id, 
    slice_i,
    mgmt_value,
    Path,
    types=("FLAIR", "T1w", "T1wCE", "T2w"),
):
    plt.figure(figsize=(16, 5))
    patient_path = os.path.join(
        Path, 
        str(brats21id).zfill(5),
    )
    for i, t in enumerate(types, 1):
        t_paths = sorted(
            glob.glob(os.path.join(patient_path, t, "*")), 
            key=lambda x: int(x[:-4].split("-")[-1]),
        )
        data = load_dicom(t_paths[int(len(t_paths) * slice_i)])
        plt.subplot(1, 4, i)
        plt.imshow(data, cmap="gray")
        plt.title(f"{t}", fontsize=16)
        plt.axis("off")

    plt.suptitle(f"MGMT_value: {mgmt_value}", fontsize=16)
    plt.show()

def load_dicom_line(path):
    t_paths = sorted(
        glob.glob(os.path.join(path, "*")), 
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    images = []
    for filename in t_paths:
        data = load_dicom(filename)
        if data.max() == 0:
            continue
        images.append(data)
        
    return images

def create_animation(ims):
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animate.animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

def copy_images(source_folder, labels_folder, images_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            if 'seg' in file:
                destination_path = os.path.join(labels_folder, os.path.relpath(source_path, source_folder))
            else:
                destination_path = os.path.join(images_folder, os.path.relpath(source_path, source_folder))

            shutil.copy(source_path, '\\'.join(destination_path.split('\\')[:-2])+'\\'+destination_path.split('\\')[-1])
    


def organize_images_by_modality(input_directory, output_directory):
    """
    Organize medical images into different folders based on their modalities.

    Args:
        input_directory (str): Path to the input directory containing the medical images.
        output_directory (str): Path to the output directory where the images will be organized.

    Returns:
        None
    """
    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # List all files in the input directory
    file_list = os.listdir(input_directory)

    # Create subdirectories for each modality in the output directory
    modalities = ['t1c', 't1n', 't2f', 't2w']
    for modality in modalities:
        modality_directory = os.path.join(output_directory, modality)
        os.makedirs(modality_directory, exist_ok=True)

    # Move files to the corresponding modality subdirectories
    for filename in file_list:
        if filename.endswith('.nii.gz'):
            # Extract the modality from the file name (assuming the format 'BraTS-GLI-XXXXX-XXX-MODALITY.nii.gz')
            modality = filename.split('-')[-1].split('.')[0]
            if modality in modalities:
                source_path = os.path.join(input_directory, filename)
                destination_path = os.path.join(output_directory, modality, filename)
                shutil.move(source_path, destination_path)

def compare_im(image1_path,image2_path):
    from PIL import Image
    # Open the images
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

# Get the sizes of the images
    size1 = image1.size
    size2 = image2.size

# Compare the sizes
    return(size1 == size2)

def get_bounding_box(ground_truth_map,num):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map == num)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 5))
  x_max = min(W, x_max + np.random.randint(0, 5))
  y_min = max(0, y_min - np.random.randint(0, 5))
  y_max = min(H, y_max + np.random.randint(0, 5))
  bbox = [x_min, y_min, x_max, y_max]
  return bbox

def all_boxes(ground_truth_map):
  unique_pix = np.unique(ground_truth_map)
  input_boxes = []
  pixels = [1,2,3] # [1,2,3] | [1,3] | [1,2] | [2,3] | [1] | [2] | [3] |
  #list(filter(lambda x: x != 0, unique_pix))
  for value in pixels:
    if value not in unique_pix:
        input_boxes.append([0,0,0,0])
    else:  
        input_boxes.append(get_bounding_box(np.array(ground_truth_map),value))
  return input_boxes

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0,0,0,0), lw=2))  

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image,cmap='gray')
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()










