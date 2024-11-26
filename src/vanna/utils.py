import hashlib
import os
import re
import uuid
from typing import Union

from .exceptions import ImproperlyConfigured, ValidationError

SEPARATOR = "\n" + 80*'=' + "\n"

def validate_config_path(path):
    if not os.path.exists(path):
        raise ImproperlyConfigured(
            f'No such configuration file: {path}'
        )

    if not os.path.isfile(path):
        raise ImproperlyConfigured(
            f'Config should be a file: {path}'
        )

    if not os.access(path, os.R_OK):
        raise ImproperlyConfigured(
            f'Cannot read the config file. Please grant read privileges: {path}'
        )


def sanitize_model_name(model_name):
    try:
        model_name = model_name.lower()

        # Replace spaces with a hyphen
        model_name = model_name.replace(" ", "-")

        if '-' in model_name:

            # remove double hyphones
            model_name = re.sub(r"-+", "-", model_name)
            if '_' in model_name:
                # If name contains both underscores and hyphen replace all underscores with hyphens
                model_name = re.sub(r'_', '-', model_name)

        # Remove special characters only allow underscore
        model_name = re.sub(r"[^a-zA-Z0-9-_]", "", model_name)

        # Remove hyphen or underscore if any at the last or first
        if model_name[-1] in ("-", "_"):
            model_name = model_name[:-1]
        if model_name[0] in ("-", "_"):
            model_name = model_name[1:]

        return model_name
    except Exception as e:
        raise ValidationError(e)


def deterministic_uuid(content: Union[str, bytes]) -> str:
    """Creates deterministic UUID on hash value of string or byte content.

    Args:
        content: String or byte representation of data.

    Returns:
        UUID of the content.
    """
    if isinstance(content, str):
        content_bytes = content.encode("utf-8")
    elif isinstance(content, bytes):
        content_bytes = content
    else:
        raise ValueError(f"Content type {type(content)} not supported !")

    hash_object = hashlib.sha256(content_bytes)
    hash_hex = hash_object.hexdigest()
    namespace = uuid.UUID("00000000-0000-0000-0000-000000000000")
    content_uuid = str(uuid.uuid5(namespace, hash_hex))

    return content_uuid

def vn_log(message: str, title: str = "", off_flag: bool = False):
    if off_flag:
        return 
    
    if message:
        msg = f"\n[( {title} )]\n{message}" if title else f"\n{message}"
        msg += SEPARATOR
    else:
        msg = f"\n[( {title} )]" + SEPARATOR if title else ""

    if msg:
        print(msg)


def strip_brackets(ddl):
    """
    This function removes square brackets from table and column names in a DDL script.
    
    Args:
        ddl (str): The DDL script containing square brackets.
    
    Returns:
        str: The DDL script with square brackets removed.
    """
    # Use regular expressions to match and replace square brackets
    pattern = r"\[([^\]]+)]"  # Match any character except ] within square brackets
    return re.sub(pattern, r"\1", ddl)

def convert_to_string_list(df):
    """
    Convert dataframe to row-data list

    Input:
        df (dataframe) - contains business terms as additional metadata documentations,
            It has 4 columns: [business_term, business_description, related_tables, related_columns]

    Returns:
        list of str
    """
    result = []
    for _, row in df.iterrows():
        formatted_string = (
            f"business_term : {row['business_term']}; "
            f"business_description : {row['business_description']}; "
            f"related_tables : {row['related_tables']}; "
            f"related_columns : {row['related_columns']}; "
        )
        result.append(formatted_string)
    return result

def remove_sql_noise(sql):
    if 'intermediate_sql' in sql or 'final_sql' in sql:
        sql = sql.replace('intermediate_sql', '').replace('final_sql', '')
    return sql 

def extract_sql(llm_response: str, **kwargs) -> str:
    """
    Extracts SQL from LLM response with flexible pattern matching.
    
    Args:
        llm_response (str): The LLM response
        **kwargs: 
            take_last (bool): Whether to take the last match instead of first
            show_sql (bool): Whether to log the extracted SQL
    
    Returns:
        str: The extracted SQL query
    """
    # Preprocess
    llm_response = llm_response.replace("\\_", "_").replace("\\", "")
    
    # Try markdown code blocks first
    sql_blocks = re.findall(r"```sql\n(.*?)```", llm_response, re.IGNORECASE | re.DOTALL)
    if sql_blocks:
        sql = sql_blocks[-1] if kwargs.get('take_last') else sql_blocks[0]
        if kwargs.get('show_sql'):
            vn_log(f"Extracted SQL:\n{sql}")
        return remove_sql_noise(sql)
    
    # Try WITH/SELECT patterns
    pattern = r"(?:WITH|SELECT).*?(?=;|\[|```|$)"
    matches = re.findall(pattern, llm_response, re.IGNORECASE | re.DOTALL)
    if matches:
        sql = matches[-1] if kwargs.get('take_last') else matches[0]
        if kwargs.get('show_sql'):
            vn_log(f"Extracted SQL:\n{sql}")
        return remove_sql_noise(sql)
    
    return llm_response

def convert_to_string_list(df):
    """ convert dataframe to row data list
    """
    result = []
    for _, row in df.iterrows():
        formatted_string = (
            f"business_term : {row.get('business_term', '')}"
            f"business_description : {row.get('business_description', '')}"
            f"related_tables : {row.get('related_tables', '')}"
            f"related_columns : {row.get('related_columns', '')}"
        )
        result.append(formatted_string)
    return result