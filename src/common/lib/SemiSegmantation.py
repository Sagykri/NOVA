import os
import sys
import logging
from src.common.lib.StatsLog import Stats_log
import pathlib
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import block_reduce
import cv2
import datetime
from shapely.geometry import Polygon
from src.common.lib.utils import xy_to_tuple
from src.common.lib.globals import CountsDF
from src.common.lib.DataDescriptors import Parse2Descriptors
import cellpose  # comparison only
from cellpose import models   # comparison only
bShow_Tile = False

#algo params
bDebug = True
bValidation = True # vs. cellpose
bDisplay = False  # relevant to entire image only - not tiles

###################
#cellpose params - just for the comparisn
cellpose_diameter= 60
cellpose_channels = [2,0]
cellpose_cellprob_threshold = 0
cellpose_flow_threshold = 0.4
cellpose_model_type = 'nuclei'


def DrawLinesOverlay(img, Nx = 4,Ny = 4):

    color = (128, 128, 128) #gray
    thickness = 1
    [Lx,Ly] = img.shape
    Sx = int(np.floor(Lx/Nx))
    Sy = int(np.floor(Ly/Ny))

    for i in range(1,Nx):
        start_point = (i*Sx, 0)
        end_point = (i*Sx, Ly)
        img = cv2.line(img, start_point, end_point, color, thickness)

    for i in range(1, Ny):
        start_point = (0, i * Sy)
        end_point = (Lx, i * Sx)
        img = cv2.line(img, start_point, end_point, color=color, thickness=3)

    return img

def MarkValidTiles(img, ValidTilesIds, Nx = 4,Ny = 4):

    color = (255, 0, 0) #red
    thickness = 2
    [Lx,Ly] = img.shape
    Sx = int(np.floor(Lx/Nx))
    Sy = int(np.floor(Ly/Ny))

    TileNumber = 0
    #allegly x must be first since . but no the order is row wise
    # i.e.    0 1 2 3
    #         4 5 6 7

    for j in range(0, Ny):
        for i in range(0,Nx):
            if not(TileNumber in ValidTilesIds):
                start_point = (i*Sx, j* Sy)
                end_point = ((i+1)*Sx, (j+1)*Sy)
                start_point2 = ((i+1) * Sx, j * Sy)
                end_point2 = (i * Sx, (j + 1) * Sy)
                img = cv2.line(img, start_point, end_point, color, thickness)
                img = cv2.line(img, start_point2, end_point2, color, thickness)
            TileNumber += 1

    return img




def ShowComparison(img_nucleus, clean_img, masks_cellpose, TimesDic, bDisplay, NewSegVsCellpose_FileName):
    fig, axes = plt.subplots(1, 3, figsize=(40, 40), sharex=True, sharey=True)
    ax = axes.ravel()

    img_nucleus = DrawLinesOverlay(img_nucleus)
    clean_img = DrawLinesOverlay(clean_img)
    masks_cellpose = DrawLinesOverlay(masks_cellpose)
    Cellpose_Bin = 128*(masks_cellpose > 0)
    fontsize = 60
    ax[0].imshow(img_nucleus)
    ax[0].set_title("Orig",  fontsize = fontsize)

    ax[1].imshow(clean_img)
    ax[1].set_title("New G SemiSegm. mSec: " + str((TimesDic["TotalTime"])),  fontsize = fontsize)

    ax[2].imshow(Cellpose_Bin)
    ax[2].set_title("Cellpose.Entire Image",  fontsize = fontsize)

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    # plt.show()
    plt.savefig(NewSegVsCellpose_FileName)
    if bDisplay:
        plt.show()
    plt.close()

def ShowComparison_Tiles(img, SemiSegmentation_clean_img,SemiSegmentation_ValidTilesIds, Cellpose_ValidTilesIds, \
                         SemiSegmentation_TimesDic, Cellpose_Filter_AllTiles_Time, NewSegVsCellpose_FileName):


    #####################################
    #Do global Cellpose segmentation - just to get the overall binary image - easier to comapre on more than tile level
    #valid non valid tiles are input and done in a regular way
    Helper = img[:, :, 1]  # IS todo check with Sagy how do i state DAPI channel
    Nuclie_img = Helper.reshape(Helper.shape[0], Helper.shape[1])

    model = models.Cellpose(gpu=True, model_type=cellpose_model_type)

    masks_cellpose, flows, styles, diams = model.eval(Nuclie_img, diameter=cellpose_diameter,
                                                      channels=cellpose_channels,
                                                      cellprob_threshold=cellpose_cellprob_threshold,
                                                      flow_threshold=cellpose_flow_threshold)

    ##########################################
    # prepare visual comparisn

    SemiSegmentation_clean_img = DrawLinesOverlay(SemiSegmentation_clean_img)
    #masks_cellpose = masks_cellpose > 0
    masks_cellpose = DrawLinesOverlay(masks_cellpose)
    #Nuclie_img = DrawLinesOverlay(Nuclie_img)

    SemiSegmentation_clean_img = MarkValidTiles(SemiSegmentation_clean_img, ValidTilesIds = SemiSegmentation_ValidTilesIds)
    masks_cellpose = MarkValidTiles(masks_cellpose, ValidTilesIds=Cellpose_ValidTilesIds)
    SemiSegmentation_clean_img = 255*(SemiSegmentation_clean_img > 0)
    masks_cellpose = 255 * (masks_cellpose > 0)
    fontsize = 40
    fig, axes = plt.subplots(1, 3, figsize=(40, 40), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(Nuclie_img)
    ax[0].set_title("Orig", fontsize = fontsize)

    ax[1].imshow(SemiSegmentation_clean_img, cmap=plt.cm.gray)
    ax[1].set_title("New G SemiSegm. Time (Sec): {:.3f}".format(SemiSegmentation_TimesDic["TotalTime"]) , fontsize = fontsize )

    ax[2].imshow(masks_cellpose, cmap=plt.cm.gray)
    ax[2].set_title("Cellpose.Time (Sec): {:.3f}".format(Cellpose_Filter_AllTiles_Time) ,  fontsize = fontsize)

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()
    # plt.show()
    plt.savefig(NewSegVsCellpose_FileName)
    plt.close()

class SemiSegmentation(object):

    kernel_e = np.ones((3, 3), np.uint8)  # kernel = np.array([[-1, -1, -1], [-1, 25, -1], [-1, -1, -1]])
    ErodeNum = 0
    DilateNun = 0
    MinBlobArea = 0

    def __init__(self, ErodeNum, DilateNun, MinBlobArea):
        self.ErodeNum = ErodeNum
        self.DilateNun = DilateNun
        self.MinBlobArea = MinBlobArea

    def Segment(self, img):

        Ret, img_T = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img_T = img_T.astype('uint8')
        return img_T

    def Clean(self, ImgGry):

        morph_temp = cv2.erode(ImgGry, self.kernel_e, iterations=self.ErodeNum)
        img_morph = cv2.dilate(morph_temp, self.kernel_e, iterations=self.DilateNun)
        return img_morph, morph_temp

    def CleanSmallBlobs(self, im):
        # find all of the connected components (white blobs in your image).
        # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(im)
        # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
        # here, we're interested only in the size of the blobs, contained in the last column of stats.
        sizes = stats[:, -1]
        # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
        # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
        sizes = sizes[1:]
        nb_blobs -= 1
        NewBlobsCount = 0

        # output image with only the kept components
        im_result = np.zeros((im.shape))
        im_debris = np.zeros((im.shape))
        # for every component in the image, keep it only if it's above min_size
        for blob in range(nb_blobs):
            if sizes[blob] >= self.MinBlobArea:
                # see description of im_with_separated_blobs above
                im_result[im_with_separated_blobs == blob + 1] = blob + 1  # not 255 - since we must create blobs map - to be used by outlines_list
                NewBlobsCount +=1
            else:
                im_debris[im_with_separated_blobs == blob + 1] = blob + 1  #IS todo remove this if not debug

        print("Was {0}. Now {1}. removed {2}".format(nb_blobs, NewBlobsCount, nb_blobs-NewBlobsCount))
        return im_result, im_debris, NewBlobsCount, (nb_blobs-NewBlobsCount)

    def ShowSteps(self,img_nucleus, img_T, morph_temp, img_morph, im_debris, clean_img, TimesDic, Count, bDisplay, SegSteps_FileName):
        fig, axes = plt.subplots(2, 3, figsize=(12, 12), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(img_nucleus)
        ax[0].set_title("Orig. Overall Seg mSec: " + str(round(TimesDic["TotalTime"] * 1000)))

        ax[1].imshow(img_T)
        ax[1].set_title("Otsu. mSec: " + str(round(TimesDic["Time_Otsu"] * 1000)))

        ax[2].imshow(morph_temp)
        ax[2].set_title("eroded.")

        ax[3].imshow(img_morph, cmap=plt.cm.gray)
        ax[3].set_title("dilated.All morph mSec: " + str(round(TimesDic["Time_morph"] * 1000)))

        ax[4].imshow(im_debris, cmap=plt.cm.gray)
        ax[4].set_title("debris only. #Obj: " + str(Count["Debris"]))

        ax[5].imshow(clean_img, cmap=plt.cm.gray)
        ax[5].set_title("cleaned. mSec: {0}. #obj: {1}".format(str(round(TimesDic["Time_Clean"] * 1000)), Count["NewBlobs"]))

        for a in ax.ravel():
            a.axis('off')

        fig.tight_layout()
        # plt.show()
        plt.savefig(SegSteps_FileName)
        if bDisplay:
            plt.show()
        plt.close()

def PerformSemiSegmentation(img_nucleus, save_path, SemiSig_ERODE_NUM,SemiSig_DILATE_NUM, SemiSig_MIN_BLOB_AREA):

    Segmentor = SemiSegmentation(ErodeNum = SemiSig_ERODE_NUM, DilateNun = SemiSig_DILATE_NUM, MinBlobArea = SemiSig_MIN_BLOB_AREA)

    TimesDic= {"Time_Otsu":-1, "Time_morph":-1, "Time_Clean":-1, "TotalTime": -1}
    Count = {"NewBlobs":-1,"Debris":-1}

    # otsu
    start_time = datetime.datetime.now()
    img_T = Segmentor.Segment(img_nucleus)
    TimesDic["Time_Otsu"] = (datetime.datetime.now() - start_time).total_seconds()
    print( TimesDic["Time_Otsu"])

    #clean morph
    start_time = datetime.datetime.now()
    img_morph, morph_temp = Segmentor.Clean(img_T)
    TimesDic["Time_morph"]  = (datetime.datetime.now() - start_time).total_seconds()
    print('Morph time:'+str(TimesDic["Time_morph"]))

    # clean debris than counter
    start_time = datetime.datetime.now()

    clean_img, im_debris, Count["NewBlobs"], Count["Debris"] = Segmentor.CleanSmallBlobs(img_morph)

    TimesDic["Time_Clean"] = (datetime.datetime.now() - start_time).total_seconds()

    TimesDic["TotalTime"]  = TimesDic["Time_Otsu"]+TimesDic["Time_morph"]+TimesDic["Time_Clean"]
    print('overall time.new:' + str( TimesDic["TotalTime"]))

    if bDebug:

        FileName = Path(save_path).name
        FilePath = Path(save_path).parent
        Helper = FilePath.parts
        SignificantPath = ''
        for i in Helper[-6:]:
            SignificantPath += '_' + i
        # print(SignificantPath)
        SegSteps_FileName = os.path.join(FilePath, "SegmentationsSteps" + SignificantPath + FileName)
        #NewSegVsCellpose_FileName = os.path.join(dir_path_output, "NewSegVsCellpose" + SignificantPath + FileName)
        Segmentor.ShowSteps(img_nucleus, img_T, morph_temp, img_morph, im_debris, clean_img, TimesDic, Count, bDisplay,SegSteps_FileName)

    return  clean_img, TimesDic


def outlines_list(masks):  # copied from  cellpose.utils.outlines_list(masks)
    """ get outlines of masks as a list to loop over for plotting """
    outpix = []
    for n in np.unique(masks)[1:]:
        mn = masks == n
        if mn.sum() > 0:
            contours = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
            contours = contours[-2]
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix) > 4:
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0, 2)))
    return outpix





def  SemiSegmentation_filter_invalid_tiles(file_name, tiles , min_edge_distance ,  tile_w,tile_h):

    """
       Filter invalid tiles (leave only tiles with #nuclues (not touching the edges) == 1)
       """

    image_processed_tiles_passed = []
    n_tiles = tiles.shape[0]  # IS - in debug change here
    ValidTilesIds = []
    #FilterAllTiles_Time = 0
    #CellposeAllTiles_Time = 0
    if n_tiles > 0:  # IS
        cells_count = np.zeros([n_tiles], dtype=np.int16)  # IS

    for i in range(n_tiles):
        tile = tiles[i]

        # Nuclues seg
        logging.info(f"[{file_name}] Tile number {i} out of {n_tiles}")
        logging.info(f"[{file_name}] Segmenting nuclues")

        """
        Filter tiles with no nuclues
        """
        if bShow_Tile:
            plt.imshow(tile)

        outlines = outlines_list(tile)
        polys_nuclei = [Polygon(xy_to_tuple(o)) for o in outlines]

        # Build polygon of image's edges
        img_edges = Polygon([[min_edge_distance, min_edge_distance], \
                             [min_edge_distance, tile_h - min_edge_distance], \
                             [tile_w - min_edge_distance, tile_h - min_edge_distance], \
                             [tile_w - min_edge_distance, min_edge_distance]])

        # Is there any nuclues inside the image boundries?
        is_valid = any([p.covered_by(img_edges) for p in polys_nuclei])

        #IS - Future TODO add num of thoucing polys
        #####################################################################
        ############# 210722: New constraint - only 1-5 nuclei per tile #####
        is_valid = is_valid and (len(polys_nuclei) >= 1 and len(polys_nuclei) <= 5)
        #####################################################################

        if is_valid:
            image_processed_tiles_passed.append(tile)    # IS todo check - probably not needed - since we have Tiles ids
            ValidTilesIds.append(i)
            cells_count[i] = len(polys_nuclei)  # IS

    Overall_cells_count = np.sum(cells_count)
    TaggedDataDescriptors = Parse2Descriptors(file_name)
    Stats_log.line(f"[{TaggedDataDescriptors}] @STAT: SemiSeg Overall_cells_count: [{Overall_cells_count}]")
    #Stats_log.line(f"[{TaggedDataDescriptors}] @RunTime: FilterAllTiles_Time: [{FilterAllTiles_Time}] sec")
    #Stats_log.line(f"[{TaggedDataDescriptors}] @RunTime: CellposeAllTiles_Time: [{CellposeAllTiles_Time}] sec")
    Stats_log.vector('SemiSeg_cell_count', cells_count)
    CountsDF.AddLine(file_name, cells_count)  ## save counts as dataframe

    if len(image_processed_tiles_passed) == 0:
        logging.info(f"Nothing is valid (total: {n_tiles})")

        return np.array(image_processed_tiles_passed), ValidTilesIds

    image_processed_tiles_passed = np.stack(image_processed_tiles_passed, axis=-1)
    image_processed_tiles_passed = np.moveaxis(image_processed_tiles_passed, -1, 0)

    logging.info(f"#ALL {n_tiles}, #Passed {image_processed_tiles_passed.shape[0]}")

    return image_processed_tiles_passed, ValidTilesIds


