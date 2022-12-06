import csv
import json
from typing import List, Dict, Any, Optional


class Loader:
    
    @staticmethod
    def load(fpath: str) -> List[Dict[str, Any]]:
        raise NotImplementedError


class Writer:
    
    @staticmethod
    def write(entries: List[Any], fpath: str, header: Optional[List[str]]) -> None:
        raise NotImplementedError


class JSONLLoader(Loader):
    
    @staticmethod
    def load(fpath: str) -> List[Dict[str, Any]]:
        entries = []
        with open(fpath) as fin:
            for line in fin:
                entries.append(json.loads(line))
        return entries


class JSONLWriter(Writer):
    
    @staticmethod
    def write(entries: List[Dict[str, Any]], fpath: str) -> None:
        with open(fpath, 'w') as fout:
            for entry in entries:
                fout.write(json.dumps(entry) + '\n')


class EDOSCSVLoader(Loader):
    
    @staticmethod
    def load(fpath: str) -> List[Dict[str, Any]]:
        entries = []
        with open(fpath) as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                entries.append(row)
        return entries


class EDOSCSVWriter(Writer):

    @staticmethod
    def write(entries: List[Dict[str, Any]], fpath: str, header: List[str]) -> None:
        with open(fpath, 'w') as fout:
            writer = csv.DictWriter(fout, fieldnames=header)
            writer.writeheader()
            for entry in entries:
                writer.writerow(entry)


LOADERS = {
    'EDOSCSVLoader': EDOSCSVLoader,
    'JSONLLoader': JSONLLoader
}

WRITERS = {
    'EDOSCSVWriter': EDOSCSVWriter,
    'JSONLWriter': JSONLWriter
}
