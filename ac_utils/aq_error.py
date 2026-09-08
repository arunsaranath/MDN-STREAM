# -*- coding: utf-8 -*-
import psycopg2
import xml.etree.ElementTree as ET

def Error_Handler(func):
    def Inner_Function(*args, **kwargs):
        name = func.__name__
        try:
            return func(*args, **kwargs)
        
        #Basic type error
        except TypeError:
            print(f"Type error raised in {name}() -~- Unknown reason for this error")
            raise
            
        #Case by case asserstion statements
        except AssertionError as e:
            if str(e) == "no files to zip":
                print(f"Assertion error raised in {name}() -~- No output images were found to zip and transfer")
            elif str(e) == "no luts":
                print(f"Assertion error raised in {name}() -~- Rayleigh LUTs were loaded improperly")
            elif "No ancillary results retrieved for" in str(e):
                print(f"Assertion error raised in {name}() -~- No ancillary files for the specific date and time of this scene were found in the database, please ensure they are downloaded and re-run this code")
            elif str(e) == "mismatched angle image count":
                print(f"Assertion error raised in {name}() -~- The number of viewing angle images loaded does not match the number of rhot bands loaded")
            elif str(e) == "empty rhot list":
                print(f"Assertion error raised in {name}() -~- No rhot images were found in the location specified")
            elif str(e) == "no input dir":
                print(f"Assertion error raised in {name}() -~- No input directory exists at the location passed as kwarg by user")
            else:
                print(f"Assertion error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #Variable referenced before assignment
        except UnboundLocalError:
            if name == "run_MDN":
                print(f"Undefined variable error raised in {name}() -~- This usually indicates that all pixels in the rho_rc image were masked out by the threshold")
            else:
                print(f"Undefined variable error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #OS failed to handle a file usually
        except OSError:
            if name in ["zip_and_transfer_output","read_sentinel_images","read_landsat_images","read_ancillary"]:
                print(f"Operating System error raised in {name}() -~- This is likely due to the known issue in Ceph file transfers over 1GB, results may vary if re-run")
            else:
                print(f"Operating System error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #Database error
        except psycopg2.DatabaseError:
            print(f"Database error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #Index error
        except IndexError:
            if name == "load_gtifs":
                print(f"Index error raised in {name}() -~- An expected input file is missing, it's possible that a re-download could resolve this")
            else:
                print(f"Index error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #Value error
        except ValueError:
            if name =="rayleigh_correction":
                print(f"Value error raised in {name}() -~- This is likely due to erroneous angle values")
            else:
                print(f"Value error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #Key error
        except KeyError:
            if name == "run_MDN":
                print(f"Key error raised in {name}() -~- This probably means that the previous stage of processing did not complete for this scene")
            else:
                print(f"Key error raised in {name}() -~- Unknown reason for this error")
            raise
        
        #XML parse error
        except ET.ParseError:
            if name in ["load_landsat_meta","load_sentinel_meta"]:
                print(f"XML Parsing error raised in {name}() -~- Corrupt or incomplete XML metadata file, please re-download")
            else:
                print(f"XML Parsing error raised in {name}() -~- Unknown reason for this error")
            raise
        
        
    return Inner_Function

