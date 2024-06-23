import re

class Subtitle():
    def __init__(self,ext="srt"):
        sub_dict = {
            "srt":{
                "coma": ",",
                "header": "",
                "format": lambda i,segment : f"{i + 1}\n{self.timeformat(segment['timestamp'][0])} --> {self.timeformat(segment['timestamp'][1] if segment['timestamp'][1] != None else segment['timestamp'][0])}\n{segment['text']}\n\n",
            },
            "vtt":{
                "coma": ".",
                "header": "WebVTT\n\n",
                "format": lambda i,segment : f"{self.timeformat(segment['timestamp'][0])} --> {self.timeformat(segment['timestamp'][1] if segment['timestamp'][1] != None else segment['timestamp'][0])}\n{segment['text']}\n\n",
            },
            "txt":{
                "coma": "",
                "header": "",
                "format": lambda i,segment : f"{segment['text']}\n",
            },
        }

        self.ext = ext
        self.coma = sub_dict[ext]["coma"]
        self.header = sub_dict[ext]["header"]
        self.format = sub_dict[ext]["format"]

    def timeformat(self,time):
        hours = time // 3600
        minutes = (time - hours * 3600) // 60
        seconds = time - hours * 3600 - minutes * 60
        milliseconds = (time - int(time)) * 1000
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}{self.coma}{int(milliseconds):03d}"
    
    def get_subtitle(self,segments):
        output = self.header
        for i, segment in enumerate(segments):
            if segment['text'].startswith(' '):
                segment['text'] = segment['text'][1:]
            try:
                output += self.format(i,segment)
            except Exception as e:
                print(e,segment)
            
        return output
    
    def write_subtitle(self, segments, output_file):
        output_file += "."+self.ext
        subtitle = self.get_subtitle(segments)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(subtitle)
