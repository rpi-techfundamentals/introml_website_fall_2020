import yaml
import pandas as pd
import subprocess
import numpy as np
from tabulate import tabulate
import ruamel.yaml
DATE='Date'
DUE='Due'

def load_yaml_file(file):
    """
    Loads a yaml file from file system.
    @param file Path to file to be loaded.
    """
    try:
        with open(file, 'r') as yaml:
            kwargs = ruamel.yaml.round_trip_load(yaml, preserve_quotes=True)
        return kwargs
    except subprocess.CalledProcessError as e:
        print("error")
    return(e.output.decode("utf-8"))

def update_yaml_file(file, kwargs):
    """
    Updates a yaml file.
    @param kwargs dictionary.
    """
    print("Updating the file: " + str(file))
    try:
        ruamel.yaml.round_trip_dump(kwargs, open(file, 'w'))
    except subprocess.CalledProcessError as e:
        print("error: " + e)



# def load_yaml_file(file):
#     """
#     Loads a yaml file from file system.
#     @param file Path to file to be loaded.
#     """
#     try:
#         with open( file ) as f:
#             cf = yaml.safe_load(f)
#         return cf
#     except subprocess.CalledProcessError as e:
#         print("error")
#         return(e.output.decode("utf-8"))
#
# def update_yaml_file(file, data):
#     """
#     Updates a yaml file.
#     @param kwargs dictionary.
#     """
#     print("Updating the file: " + str(file))
#     try:
#         with open(file, 'w') as outfile:
#             yaml.dump(data, outfile, default_flow_style=False)
#
#     except subprocess.CalledProcessError as e:
#         print("error: " + e)

def create_md_title(title, content=""):
    s = title
    separator = "\n============================\n\n"
    return s+separator+content+"\n"

def pandas_to_md(df, file, title,  include, header="",footer=""):
    if DATE in df.columns:
        #if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[DATE]):
        print("Converting datetime to ")
        df[DATE]=df[DATE].dt.strftime('%m/%d')
    if DUE in df.columns:
        #if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[DATE]):
        print("Converting datetime to ")
        df[DUE]=df[DUE].dt.strftime('%m/%d')
    #Deal with NA, float, str
    df=df.fillna(-99)

    float_cols = df.select_dtypes(include=[np.float]).columns
    df[float_cols]=df[float_cols].astype(int)
    df=df.astype(str)
    df=df.replace('-99', ' ')
    md_title=create_md_title(title, header)

    pd.set_option('precision', 0)

    table=df.loc[:,include].to_markdown(tablefmt="pipe", headers="keys", index="never")

    #table=tabulate(df, tablefmt="pipe", headers="keys")
    output= md_title+table+footer
    print("Outputting file:", file)
    with open(file, "w") as text_file:
        text_file.write(output)
    return df

def add_row_md(md_file, title, df):
    md_file =md_file+'\n## '+title+'\n\n'
    md_file =md_file+df.iloc[:,0:-1].to_markdown(tablefmt="pipe", headers="keys", index="never")
    md_file =md_file+'\n\n'
    return md_file


def generate_sessions(config, toc, toc_part, schedule, path, content, keys):
    toc[toc_part]['chapters']=[] #zero out the sessions
    for index, row in schedule.iterrows():
        if row['Publish']=='1':
            toc[toc_part]['chapters'].append({'file': 'sessions/session'+row['Session']})
            md_file=create_md_title(row['Topic']+' ('+row['Date'] +')', row['Summary'])
            for key in keys:
                content[key]=content[key].astype(str)
                selection= content[key].loc[content[key]['Session']==row['Session'],:]
                if len(selection)>0:
                    md_file=add_row_md(md_file, key, selection)
            file='session'+row['Session']+'.md'
            print("Outputting ", file)
            with open(path / file, "w") as text_file:
                text_file.write(md_file)
    return toc

def link_generator(df, target,repo_url,link_name):
    for index, row in df.iterrows():
        if row['Type']=='Link':
            df.loc[index,target]=row[target]+" ["+link_name+"]("+row['Location']+")"
        elif row['Type']=='File':
            df.loc[index,target]=row[target]+" ["+link_name+"]("+repo_url+"/raw/master/"+row['Location']+")"
        elif row['Type'] in ['Notebook', 'Markdown']:
            df.loc[index,target]=row[target]+" ["+link_name+"](../"+row['Location']+")"
        #elif row['Type']=='Youtube'
    df.drop(columns=['Location'],inplace=True)
    return df

def create_syllabus(df, item, message, path,repo_url):
    location=df.loc[item,'Location']
    print(location)
    message=message+"\n[Syllabus]("+repo_url+"/raw/master/"+location+")"
    message=create_md_title("Syllabus", content=message)
    print("Outputting ", path)
    with open(path , "w") as text_file:
        text_file.write(message)
