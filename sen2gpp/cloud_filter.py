# Cloud filter code
        
# aoi =   ee.Geometry.Polygon(
#         [[[3.3674943040548055, 51.359071544700114],
#           [3.4968603449010263, 51.39054553905237],
#           [3.527367125474586, 51.39581920664727],
#           [3.670879921841975, 51.35767359436224],
#           [3.7409497893414647, 51.33581817395117],
#           [3.815078709914628, 51.31906604991423],
#           [3.8878643736126905, 51.33236872551182],
#           [3.9197849791433264, 51.34888847215252],
#           [3.9640840870638967, 51.35553002409429],
#           [3.979880982341294, 51.396898475788696],
#           [4.009425069876966, 51.39293144551682],
#           [4.0313982771382, 51.36951693587628],
#           [4.045295804863651, 51.36048921026879],
#           [4.077725361073385, 51.35574425812928],
#           [4.112588118208083, 51.33440973916359],
#           [4.154975784628443, 51.322499771616165],
#           [4.187247967725903, 51.326795040080256],
#           [4.225701773401491, 51.345236996729376],
#           [4.250775094567714, 51.328711769690194],
#           [4.260212939805279, 51.30800430439589],
#           [4.264586641388184, 51.296362892681806],
#           [4.284410816396309, 51.287298380311974],
#           [4.304146670462035, 51.28278775641236],
#           [4.287157264809203, 51.27054663333748],
#           [4.290591407536915, 51.25593799179998],
#           [4.315137652913358, 51.25550857645474],
#           [4.322090952809781, 51.25615300512224],
#           [4.320804797944691, 51.26023514136016],
#           [4.318745324862818, 51.26732461268192],
#           [4.3362548116103605, 51.27999698960498],
#           [4.301237062352579, 51.30790647987561],
#           [4.289563488484963, 51.329365348952834],
#           [4.289568568751603, 51.35166825523348],
#           [4.265529744366401, 51.36582097861157],
#           [4.253022235160679, 51.392603896222106],
#           [4.222641277896252, 51.41081363019624],
#           [4.150539598864447, 51.40310380226782],
#           [4.086345821513098, 51.41574274272513],
#           [4.0585284342655426, 51.43350758964345],
#           [3.9651419492670676, 51.4681695945518],
#           [3.932562829280956, 51.464310627451844],
#           [3.8931434432112333, 51.433066911049075],
#           [3.8843045898333046, 51.418082780612096],
#           [3.8720380099024, 51.40480816250277],
#           [3.8429158793981024, 51.39710657113798],
#           [3.811588132902832, 51.39695024297558],
#           [3.7837022971546395, 51.4139279223521],
#           [3.733421250557379, 51.430728287070565],
#           [3.7315267147172637, 51.47361681708443],
#           [3.707644724532746, 51.4831376810955],
#           [3.6207461833676855, 51.4583549482181],
#           [3.5751185860699843, 51.455334480642804],
#           [3.555140536102988, 51.459781561657806],
#           [3.5365291299149164, 51.471070434874655],
#           [3.4931083803428638, 51.50559416472176],
#           [3.444652617212295, 51.54040182900938],
#           [3.1953944575570032, 51.53912052573608],
#           [3.196767913170066, 51.33108127220712]]])

# # Define polygon geometries
# # Define points geometries
# point_1 = ee.Geometry.Point([4.250932603,	51.35118367])
# point_2 = ee.Geometry.Point([4.1028322,	51.36697851])
# point_3 = ee.Geometry.Point([4.21026091,	51.40004802])
# point_4 = ee.Geometry.Point([4.014388534,	51.43701537])
# point_5 = ee.Geometry.Point([3.788685156,	51.35129967])
# point_6 = ee.Geometry.Point([3.553065957,	51.45045285])

# # Define visualization parameters
# visualization = {
#     'bands': ['B4', 'B3', 'B2'],
#     'min': 0.0,
#     'max': 0.3,
# }
# bitvisualization = {
#     'min': 0,
#     'max': 1,
#     'palette': ['white', 'blue'],
# }
# bitvisualizationb2 = {
#     'min': 0,
#     'max': 1,
#     'palette': ['white', 'orange'],
# }


# # Describe functions
# # Function to scale the reflectance bands
# def apply_scale_factors_s2(image):
#     optical_bands = image.select(['B.']).divide(10000)
#     thermal_bands = image.select(['B.*']).divide(10000)
#     return image.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)

# # Function to create mask with cirrus clouds and cirrus pixels
# def extract_bit_s2_10_11(image):
#     bit_position_clouds = 10
#     bit_position_cirrus = 11

#     # Bits 10 and 11 are clouds and cirrus, respectively.
#     cloud_bit_mask = 1 << bit_position_clouds
#     cirrus_bit_mask = 1 << bit_position_cirrus

#     mask_clouds = image.bitwiseAnd(cloud_bit_mask).rightShift(bit_position_clouds)
#     mask_cirrus = image.bitwiseAnd(cirrus_bit_mask).rightShift(bit_position_cirrus)
#     mask = mask_clouds.add(mask_cirrus)
#     return mask

# # Function to mask pixels with high reflectance in the blue (B2) band. The function creates a QA band
# def b2_mask(image):
#     B2Threshold = 0.2
#     B2Mask = image.select('B2').gt(B2Threshold)
#     return image.addBands(B2Mask.rename('B2Mask'))

# # Function to create a band with ones
# def make_ones(image):
#     # Create a band with ones
#     ones_band = image.select('B2').divide(image.select('B2'))
#     return image.addBands(ones_band.rename('Ones'))

# # Function to calculate area
# def get_area(img):
#     cloud_area = make_ones(img).select('Ones').multiply(ee.Image.pixelArea()) \
#         .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=30).values().get(0)
#     return img.set('area_image', ee.Number(cloud_area))

# # Function to get local cloud percentage with QA band
# def get_local_cloud_percentage(img):
#     cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
#         .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
#     return img.set('local_cloud_percentage', ee.Number(cloud_area).divide(aoi.area()).multiply(100).round())

# # Function to get local cloud percentage with QA and area of image band
# def get_local_cloud_percentage_area_image(img):
#     area_image = img.get('area_image')
#     cloud_area = extract_bit_s2_10_11(img.select('QA60')).multiply(ee.Image.pixelArea()) \
#         .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
#     return img.set('local_cloud_percentage_ai', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

# # Function to get local cloud percentage with B2 and area of image band
# def get_local_cloud_percentage_area_image_b2(img):
#     area_image = img.get('area_image')
#     cloud_area = b2_mask(img).select('B2Mask').multiply(ee.Image.pixelArea()) \
#         .reduceRegion(reducer=ee.Reducer.sum(), geometry=aoi, scale=60).values().get(0)
#     return img.set('local_cloud_percentage_ai_b2', ee.Number(cloud_area).divide(ee.Number(area_image)).multiply(100).round())

# def add_ndvi(image):
#     # Calculate NDVI
#     ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
#     return image.addBands(ndvi)


# # Get Sentinel 2 collection
# s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate('2015-01-01', '2024-12-31').filterBounds(aoi).filter(ee.Filter.stringStartsWith('GRANULE_ID', 'L1C'))
# s2 = s2.filter(ee.Filter.eq('MGRS_TILE', '31UET')).map(lambda image: image.clip(aoi)).map(apply_scale_factors_s2).map(add_ndvi)

# # Processing
# # Mask with band 2
# extractedBitB2 = s2.select('B2').map(b2_mask)
# # Mask with QA60 band
# extractedBit = s2.select('QA60').map(extract_bit_s2_10_11)
# # Band with ones
# extractedBitones = s2.map(make_ones)
# # Calculate area
# s2 = s2.map(get_area)
# # Calculate local cloud percentage with QA band
# s2 = s2.map(get_local_cloud_percentage)
# # Calculate local cloud percentage with QA band and area image band
# s2 = s2.map(get_local_cloud_percentage_area_image)
# # Calculate local cloud percentage with B2 band and area image band
# s2 = s2.map(get_local_cloud_percentage_area_image_b2)
# # Filter images
# LOCAL_CLOUD_THRESH = 30
# s2_filtered = s2.filter(ee.Filter.lte('local_cloud_percentage_ai', LOCAL_CLOUD_THRESH))
# s2_filtered = s2_filtered.filter(ee.Filter.lte('local_cloud_percentage_ai_b2', LOCAL_CLOUD_THRESH))

# # Show messages
# print('The original size of the collection is', s2.size().getInfo())
# # print(s2.first().getInfo())
# print('The filtered size of the collection is', s2_filtered.size().getInfo())


# def local_cloud_filter(s2, aoi, LOCAL_CLOUD_THRESH):

#     s2 = s2.filterBounds(aoi).map(lambda image: image.clip(aoi)).map(apply_scale_factors_s2).map(add_ndvi)
    
#     # Processing
#     # Mask with band 2
#     extractedBitB2 = s2.select('B2').map(b2_mask)
#     # Mask with QA60 band
#     extractedBit = s2.select('QA60').map(extract_bit_s2_10_11)
#     # Band with ones
#     extractedBitones = s2.map(make_ones)
#     # Calculate area
#     s2 = s2.map(get_area)
#     # Calculate local cloud percentage with QA band
#     s2 = s2.map(get_local_cloud_percentage)
#     # Calculate local cloud percentage with QA band and area image band
#     s2 = s2.map(get_local_cloud_percentage_area_image)
#     # Calculate local cloud percentage with B2 band and area image band
#     s2 = s2.map(get_local_cloud_percentage_area_image_b2)
#     # Filter images
#     # LOCAL_CLOUD_THRESH = 30
#     s2_filtered = s2.filter(ee.Filter.lte('local_cloud_percentage_ai', LOCAL_CLOUD_THRESH))
#     s2_filtered = s2_filtered.filter(ee.Filter.lte('local_cloud_percentage_ai_b2', LOCAL_CLOUD_THRESH))

#     # Show messages
#     print('The original size of the collection is', s2.size().getInfo())
#     # print(s2.first().getInfo())
#     print('The filtered size of the collection is', s2_filtered.size().getInfo())
#     print('\n')
    
#     return s2_filtered 


# # Get Sentinel 2 collection
# lon_lat         =  [-6.4345, 36.9985]
# point = ee.Geometry.Point(lon_lat)
# aoi    = point.buffer(375)

# s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate('2020-01-01', '2021-12-31').filterBounds(point)

# # apply cloud local filter
# s2_filtered = local_cloud_filter(s2, aoi, 0)