import tcod
import emoji

class Utils:
    @staticmethod
    def get_CP437_ch(idx):
        return chr(tcod.tileset.CHARMAP_CP437[idx])
    
    @staticmethod
    def get_spacer(emo, n):
        spacer = ""
        for i in range(n):
            spacer += emoji.emojize(emo)
        return spacer 