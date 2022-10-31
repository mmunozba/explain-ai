import numpy
import pandas
import logging
from io import StringIO

counter = 0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_tag_columns(tag_string_column, unique_tags = 0):
    logger.info('Getting tag list column from Tags column.')
    tag_list_column = get_tag_list_column(tag_string_column)
    logger.info(f'Head of tag_list_column: {tag_list_column.head()}')
    
    logger.info('Getting unique tags for column names.')
    if unique_tags == 0:
        unique_tags = get_unique_tags(tag_list_column)
    else:
        logger.info('Unique tags already provided.')
    
    logger.info(f'List of unique tags: {unique_tags}')
    
    logger.info('Getting rows for new tag columns.')
    tag_list_rows = get_tag_list_rows(unique_tags, tag_list_column)
    logger.info(f'tag_list_rows head: {tag_list_rows.head()}')
    
    tag_columns = pandas.DataFrame(tag_list_rows.to_numpy(),
                                   columns=[unique_tags])
    logger.info(f'tag_columns head: {tag_columns.head()}')
    
    return tag_columns


def get_tag_list_rows(unique_tags: list, tag_list_column):
    tag_list_rows = pandas.DataFrame()
    
    for index, tag_list in tag_list_column.iterrows():
        tag_list_row = []
        tag_list = tag_list.to_numpy()
        
        # Go through each column
        for unique_tag in unique_tags:
            # logger.info(f'Tag list: {tag_list}')
            # logger.info(f'Unique tag: {unique_tag}')
            # Tag is in sample
            if unique_tag in tag_list:
                tag_list_row = numpy.append(tag_list_row, True)
             # Tag is not in sample
            else:
                # logger.info(f'{unique_tag} is not in {tag_list}')
                tag_list_row = numpy.append(tag_list_row, False)
        # logger.info(f'----------------------------------------------------')
        # logger.info(f'ROW: {tag_list_row}')
        
        tag_list_rows = tag_list_rows.append(pandas.DataFrame(tag_list_row).T)
    # logger.info(f'ROWS HEAD: {tag_list_rows.head()}')
    
    return tag_list_rows


def get_tag_list_column(tag_string_column: list):
    """
    Transforms a list of tag strings into a list of lists of tags.

    Parameters
    ----------
    tag_string_column: list
        List of tag strings. (i.e. raw tags column)

    Returns
    -------
    tag_list_column: pandas.Dataframe
        List of list of tags.

    """
    global counter
    tag_list_column = pandas.DataFrame()
    
    for tag_string in tag_string_column:
        tag_list = get_tag_list(tag_string)
        
        if counter == 1:
            counter = counter + 1
            logger.info(f'Post Method - First tag list: {tag_list}')
        tag_list_column = tag_list_column.append(tag_list)
    
    return tag_list_column


def get_unique_tags(tag_list_column):
    """
    Takes a list of lists of tags and returns all unique tag names.

    Parameters
    ----------
    tagColumn: list
        List of lists of tags (i.e. tag list).

    Returns
    -------
    tags: list
        List of unique tag names.
    """
    tags = []
    
    for tag_list in tag_list_column.to_numpy():
        for tag in tag_list:
            if f'{tag}' == 'nan':
                pass
            elif tag in tags:
                pass
            else:
                tags = numpy.append(tags, tag)
                # logger.info(f'Adding {tag} to tag list.')
        
    return tags


def get_tag_list(tag_string: str, separator: str = ','):
    """
    Takes a string with tags and returns a list of the tags split by the given separator.

    Parameters
    ----------
    tagString: str
        String with tags.
    separator: str, optional
        Separator used to split the tags. The default is ','.

    Returns
    -------
    tag_list: list
        List with the tags.
    """
    global counter
    tag_list = pandas.read_csv(StringIO(tag_string.replace("\xa0", " ").replace(", ", ",")), sep=separator, header=None)
    
    if counter == 0:
        counter = counter + 1
        logger.info(f'In Method - First tag list: {tag_list.head()}')
        
    return tag_list


def transform_tags(filename: str, sheet_name: str, remove_columns: list, is_sample_set: bool = False):
    logger.info('Reading input EXCEL file.')
    input_dataframe = pandas.read_excel(filename, sheet_name=sheet_name)
    logger.info(input_dataframe.head())
    
    logger.info('Getting tags from Tags column.')
    tag_string_column = input_dataframe['Tags']
    logger.info(tag_string_column.head())
    
    logger.info('---')
    logger.info('Transforming tag string column to tag columns.')
    logger.info('---')
    if is_sample_set == True:
        input_dataframe_compl = pandas.read_excel(filename, sheet_name='Completed')
        tag_string_column_compl = input_dataframe_compl['Tags']
        tag_list_column_compl = get_tag_list_column(tag_string_column_compl)
        unique_tags = get_unique_tags(tag_list_column_compl)
        tag_columns = get_tag_columns(tag_string_column, unique_tags)
    else:
        tag_columns = get_tag_columns(tag_string_column)
    
    logger.info(f'Tag columns shape: {tag_columns.shape}')
    logger.info(f'Tag columns head: {tag_columns.head()}')
    logger.info('---')
    
    logger.info('Removing old Tags column.')
    dataframe_no_tags = input_dataframe.drop(columns=['Tags'])
    logger.info(f'Dataframe no tags shape: {dataframe_no_tags.shape}')
    logger.info('Adding new tag columns.')
    dataframe_new_tags = dataframe_no_tags.join(tag_columns)
    logger.info(f'Dataframe with new tags: {dataframe_new_tags.head()}')
    
    logger.info('Removing unwanted features.')  
    dataframe_removed_features = dataframe_new_tags.drop(columns=remove_columns)
    
    logger.info('Exporting new dataset.') 
    dataframe_removed_features.to_excel(f'{sheet_name}_tags_replaced.xlsx')
