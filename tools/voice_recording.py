"""Guided recording prompts and durable dataset saves; never starts a microphone."""

from __future__ import annotations

from datetime import datetime
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import threading
import uuid

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "recordings" / "MyVoice"
RECORD_INFO = ROOT / "recordings" / ".saved_recordings"
_SAVE_LOCK = threading.Lock()
_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

PROMPTS = [
    {
        "title": "早晨的小街",
        "instruction": "用平时说话的声音慢慢读，标点处自然换气；读错一两个字没关系。",
        "text": "清晨，我推开窗户，听见楼下传来轻轻的脚步声。街角的小店刚刚开门，老板把椅子摆在门口，又认真擦了擦桌面。一辆蓝色自行车从路边经过，车铃清脆地响了两声。我穿好外套，带着钥匙出门，准备去买一份热乎乎的早餐。今天没有特别着急的事情，可以慢慢走，也可以多看看身边的风景。阳光落在树叶上，风吹过来的时候，细小的影子就在地面摇晃。我觉得，这样普通的早晨也挺好。",
    },
    {
        "title": "和朋友商量周末",
        "instruction": "像给朋友发语音一样，问句稍微上扬，回答自然一点。",
        "text": "你周末有空吗？我们好久没有一起出去走走了。我想找个安静的地方，喝杯茶，再吃点简单的东西。如果天气不错，就去河边散步；要是下雨，我们可以找一家书店。时间不用定得太早，上午十点半怎么样？你不用带很多东西，穿一双舒服的鞋就行。对了，上次你说想看的那部电影，我已经记下名字了。我们也可以先吃午饭，再一起去看。你觉得哪个安排更合适？不用急着回答，想好了告诉我。",
    },
    {
        "title": "雨天的窗边",
        "instruction": "比平时稍慢一点，用舒服、放松的声音读，不用刻意压低嗓音。",
        "text": "下午突然下起了雨，开始只是几滴，后来雨点连成一片，敲在窗台上。我把还没看完的书放到一边，给自己倒了一杯温水。远处的楼房变得有些模糊，路上的人撑开雨伞，脚步比刚才快了许多。屋里很安静，只有时钟偶尔发出轻微的声响。我想起小时候，每逢下雨，总喜欢坐在门口看水流经过。那时候不懂什么叫忙碌，也不会担心明天的安排。现在再听见雨声，心里还是会慢慢安静下来，仿佛终于可以休息一会儿。",
    },
    {
        "title": "一个开心的消息",
        "instruction": "带一点笑意和精神，音量比前几段略大即可，不要喊叫。",
        "text": "我有一个好消息想告诉你！之前一直担心的那件事，今天终于顺利完成了。过程虽然比想象中麻烦，但最后的结果还不错。刚看到消息的时候，我忍不住笑了起来，随后又看了一遍，确认自己没有弄错。为了庆祝一下，我打算做一顿喜欢的晚饭，再听几首轻松的歌。其实让我高兴的，不只是事情成功了，还有那些愿意帮忙、认真鼓励我的人。等下次见面，我要好好说声谢谢，也希望把这份开心分享给你。",
    },
    {
        "title": "厨房里的晚饭",
        "instruction": "按自然节奏读清楚食物名称和动作，不必像播音员一样用力咬字。",
        "text": "今天晚上，我准备做一碗番茄鸡蛋面。先把番茄洗干净，切成大小差不多的小块，再把鸡蛋打进碗里，轻轻搅匀。锅热以后倒一点油，炒熟鸡蛋，接着放入番茄，等汤汁慢慢变浓。另一口锅里烧好水，把面条放进去，用筷子轻轻拨开。最后加上青菜、盐和少量胡椒，把热汤浇在面上。忙完这些，厨房里已经飘出香味。虽然不是什么复杂的菜，但坐下来吃第一口的时候，我还是觉得很满足。",
    },
    {
        "title": "出门前的提醒",
        "instruction": "像交代一件日常小事，数字和地点读清楚，后半段可以稍快一点。",
        "text": "明天我们八点二十分在车站门口见面，记得提前十分钟出门。车票、手机和身份证最好今晚就检查一遍，免得早晨临时寻找。天气预报说，上午比较凉，下午可能有太阳，所以带一件薄外套就够了。我已经查好了路线，到站后向右转，经过一家便利店，再往前走大约两百米，就能看到入口。如果你比我先到，可以在门口的长椅上等一会儿。路上不用赶，注意看车；遇到任何变化，直接给我打电话。",
    },
    {
        "title": "傍晚的散步",
        "instruction": "用平常的声音读，在逗号处短暂停顿，最后几句放松收尾。",
        "text": "吃过晚饭以后，我沿着小路往公园走。草地上的孩子还在追逐，几只小鸟停在树梢，不时抖动一下翅膀。一位老人牵着小狗慢慢经过，小狗看见落叶，就好奇地凑上去闻。天空从浅蓝变成淡淡的橙色，路灯一盏接一盏亮起来。我没有戴耳机，只是听着周围的声音，感受风从脸边吹过。有时候，放下手机走上十几分钟，就能发现一天里忽略的细节。等我转身回家，心情已经比出门时轻松了许多。",
    },
    {
        "title": "把一件小事做好",
        "instruction": "像认真分享自己的想法，语气有起伏即可，不要模仿别人的声音。",
        "text": "最近，我决定每天留一点时间学习新的东西。开始的时候，不需要制定很大的目标，也不用和别人比较。我可以读几页书，练习一道简单的菜，或者把桌面整理干净。遇到不明白的地方，就停下来查一查；做得不够好，也可以明天再试一次。我发现，真正有用的进步，常常藏在这些不起眼的小事里。只要保持一点耐心，原本陌生的事情就会慢慢熟悉。今天做完以后，给自己一点肯定，然后好好休息，明天继续往前走。",
    },
    {
        "title": "清唱入门：把元音唱长一点",
        "instruction": "不会音阶也没关系。选一个最舒服的音，把每个“啊”或“啦”唱约两三秒，中间正常换气。轻松唱，不要憋气或追求高音；也可先跳过，之后再补清唱。",
        "text": "啊—— 啊—— 啊——\n\n啦—— 啦—— 啦——\n\n呜—— 呜—— 呜——\n\n啊—— 啦—— 呜——\n\n舒服地换口气，再把上面四行唱一遍。",
    },
    {
        "title": "清唱一：风吹过窗前",
        "instruction": "没有固定曲调。用你熟悉、舒服的旋律把下面原创短句唱出来，每行末尾稍微拖长。不要播放伴奏；唱两三遍也可以，音高自然变化就好。",
        "text": "风轻轻吹过窗前，\n阳光停在我的肩。\n慢慢走过这一天，\n把微笑留在心里面。\n\n抬头看看蓝色的天，\n听见远处有人轻轻唱。\n不着急走到多远，\n这一刻就很温暖。",
    },
    {
        "title": "清唱二：晚风的小路",
        "instruction": "继续用自己的旋律唱，试着让前两行轻一些、后两行稍有力量。保持舒服的音区，不必唱得准，也不用模仿原唱。可以完整重复两三遍。",
        "text": "晚风经过安静的小路，\n带来树叶轻轻的祝福。\n我把今天的快乐记住，\n明天还要向前迈一步。\n\n星光落在远处的屋顶，\n夜色陪着我慢慢前行。\n唱一句柔软的心情，\n让声音停在你的耳边。",
    },
    {
        "title": "清唱三：换一换舒服的音高",
        "instruction": "先用平常的音高唱一遍，再稍低一点、稍高一点各唱一遍。只在舒服范围内变化，不要硬挤高音。不想唱时，也可以先把这段自然朗读下来。",
        "text": "走过晴天，也走过雨天，\n心里的歌还在耳边。\n一句问候，一张笑脸，\n都是值得珍惜的瞬间。\n\n啦啦啦，风吹向远方，\n啦啦啦，心里有微光。\n慢慢唱，不慌张，\n把自己的声音轻轻放。",
    },
    {
        "title": "舒服的低音说话",
        "kind": "reading",
        "focus": "低音清晰度",
        "instruction": "用自然、偏低但舒服的音区朗读。不要压喉咙，也不要故意把声音说得很沉；每两句换一次气，重点是低音仍然清楚。",
        "text": "夜里回到家，我先把外套挂在门边。厨房还留着一盏小灯，桌上放着一杯温水。今天的事情已经告一段落，我不用急着回答任何消息。关上门以后，房间安静下来，连自己的脚步声都听得很清楚。",
    },
    {
        "title": "明亮的高音说话",
        "kind": "reading",
        "focus": "高音放松度",
        "instruction": "把声音放在前面，像和站在稍远处的人说话，音高比平常略高即可。不要喊叫；如果高音紧张，就退回舒服音区再录一遍。",
        "text": "你好，能听见我说话吗？今天的天气真的很好！我们从这边往前走，过了那座桥就能看到开满花的院子。别担心，慢慢来就好，我会在前面等你。",
    },
    {
        "title": "由轻到强的音量渐变",
        "kind": "reading",
        "focus": "弱声到强声",
        "instruction": "每一行连续读三遍：第一遍轻声、第二遍正常、第三遍稍有力量。音高和语速尽量保持接近，只改变音量，不要用喊的方式增加力度。",
        "text": "我在这里。\n请听我说。\n这件事可以慢慢完成。\n我知道自己要去哪里。\n谢谢你一直陪着我。",
    },
    {
        "title": "元音和尾音收束",
        "kind": "reading",
        "focus": "元音延展与收尾",
        "instruction": "先清楚读字，再把“轻、边、心、见”中的元音稍微拉长，最后干净收尾，不要拖成气声。可录两遍：一遍正常，一遍稍慢。",
        "text": "今天的风很轻，轻轻吹过窗边。\n我想把这份心情留在心里。\n如果你愿意，我们明天再见。\n走到路的尽头，我会停下来听一听。",
    },
    {
        "title": "气声与实声的对比",
        "kind": "reading",
        "focus": "气声与实声",
        "instruction": "每一行先用很轻、带一点气息的声音读一次，再用清楚的正常声音读一次。气声不要压低到听不见，也不要摩擦喉咙；不舒服就只录正常版本。",
        "text": "别着急，慢慢说。\n我在听，你继续吧。\n窗外的灯还亮着。\n这一句话，我想认真告诉你。",
    },
    {
        "title": "连读和短句速度",
        "kind": "reading",
        "focus": "连读与速度变化",
        "instruction": "第一遍慢速清楚读，第二遍用自然语速连读，第三遍稍快但不能含糊。每句之间只停半拍，练习连续发音和不同速度下的稳定性。",
        "text": "如果明天有时间，我们就一起去看看。\n我已经把需要带的东西整理好了。\n先听我说完，再决定要不要改变计划。\n不用反复确认，照着自己的节奏来就好。",
    },
    {
        "title": "声母韵母清晰度",
        "kind": "reading",
        "focus": "普通话咬字",
        "instruction": "用平常音量逐行朗读，注意 z、c、s、zh、ch、sh、j、q、x 和 n、l 的区别。不要刻意播音，清楚比夸张更重要；可重复两遍。",
        "text": "早晨的紫色窗帘被风吹动。\n池边传来清脆的虫鸣声。\n小杰拿起橙色的水杯，走向安静的长廊。\n南方的柳树在细雨里轻轻摇晃。",
    },
    {
        "title": "情绪从平静到兴奋",
        "kind": "reading",
        "focus": "情绪和语调变化",
        "instruction": "同一段连续录三遍：第一遍平静，第二遍期待，第三遍明显开心但不喊叫。保留自然停顿，让情绪通过音高、速度和音量变化体现。",
        "text": "我刚刚收到一个消息，事情有了新的进展。原来我们期待的那一天，真的快要到了。等大家都准备好以后，我们就一起出发吧。",
    },
    {
        "title": "副歌前的弱声和爆发",
        "kind": "singing",
        "focus": "副歌力度与动态",
        "instruction": "用自己的旋律清唱两遍。前四行轻柔、像主歌；后四行逐渐打开，像副歌但不要喊。第二遍可以把后四行唱得更有力量，保持同一字词清楚。",
        "text": "我把声音藏在风里，\n把愿望放在夜里。\n慢慢走，不必着急，\n等天亮再回答你。\n\n我要大声唱给你听，\n让微光穿过这片安静。\n就算远方还有雨，\n我也会向前走下去。",
    },
    {
        "title": "低音到高音的换区",
        "kind": "singing",
        "focus": "音域过渡",
        "instruction": "用舒服的旋律唱下面八行。前两行偏低，中间四行逐渐升高，最后两行回到舒服位置。不要硬冲高音；换不上去时降低整段音高，完整唱完即可。",
        "text": "晚风落在肩上，\n脚步变得很轻。\n我抬头看着远方，\n听见心里的回声。\n星光越过屋顶，\n照亮没有人的小径。\n我唱完这一句，\n再安静地呼吸。",
    },
    {
        "title": "真假声和滑音连接",
        "kind": "singing",
        "focus": "真假声和滑音",
        "instruction": "先用“呜”从低到高再回到低，像轻轻的警报声那样滑上去再滑下来，做三次；然后把下面短句唱两遍。滑动要连贯，不追求音准，不要挤压喉咙。",
        "text": "呜——呜——呜——\n啊——啊——啊——\n从低处走到高处，\n再回到最舒服的地方。",
    },
    {
        "title": "长音、换气和句尾",
        "kind": "singing",
        "focus": "长音稳定与换气",
        "instruction": "用自己的旋律唱两遍。每行末尾保持一个稳定长音约三秒，提前吸气，不要唱到最后突然挤压。第二遍可把句尾收得更轻，练习自然结束。",
        "text": "我会记得这一天——\n也会记得你的笑脸——\n当风吹过很远很远——\n我的歌还在耳边——\n\n慢慢走到灯光里面——\n把心事交给夜色保管——\n明天醒来睁开双眼——\n再把新的故事唱完——",
    },
    {
        "title": "近声与远声",
        "batch": 2,
        "kind": "reading",
        "focus": "距离感和音量控制",
        "instruction": "第一遍像在近处轻声说话，第二遍像对两三米外的人说话。保持字词和音高自然，只改变投射距离和音量，不要喊。",
        "text": "我就在这里，你不用急着找我。\n如果听不清，我会再说一遍。\n窗边有一点风，门口的灯还亮着。\n我们保持自己的节奏，慢慢把事情说完。",
    },
    {
        "title": "句中换气",
        "batch": 2,
        "kind": "reading",
        "focus": "呼吸规划和连续语流",
        "instruction": "按逗号短换气，句号再完整换气。第一遍按标点读，第二遍尽量一口气读完每个短分句；不要憋气，气不够就自然停下。",
        "text": "如果今天还有时间，我们先把最重要的事情完成，再慢慢处理其他安排。\n我知道路上可能会下雨，所以带了外套，也把需要的资料放在包里。\n等你准备好了，我们就从第一步开始，不用一次把所有问题解决。",
    },
    {
        "title": "c、ch 等相近声母区分",
        "batch": 2,
        "kind": "reading",
        "focus": "c/ch、z/zh、s/sh、q/j/x、n/l 的发音区分",
        "instruction": "先逐组慢读词例，再读后面的连续句。c（擦、粗、草）舌尖靠近上齿龈并送气；ch（插、出、炒）舌尖稍向后卷，也要送气。再分清 z/zh、s/sh、q/j/x、n/l。用平常音量，舌头动作清楚但不夸张；录两遍，第二遍恢复自然语速。",
        "text": "擦窗 / 插花；粗心 / 出门；青草 / 炒菜。\n早晨 / 找人；紫色 / 纸张；三次 / 山上。\n七点 / 鸡蛋 / 西边；南方 / 蓝色；牛奶 / 流水。\n\n早晨，我在车站擦了擦桌面，又插好一束花。\n粗心的小陈出了门，才想起厨房里还炖着汤。\n紫色的车停在山脚，三只小鸟飞过树林。\n南宁来的小林拎着牛奶，沿着小路慢慢走。",
    },
    {
        "title": "开口与圆唇元音",
        "batch": 2,
        "kind": "reading",
        "focus": "口型和元音稳定",
        "instruction": "读“啊、诶、衣、哦、乌”时把口型做完整，不要咬字过度。第一遍慢速，第二遍自然连读，保持每个元音的音色稳定。",
        "text": "啊，原来你已经到了。\n这件事可以再想一想。\n衣柜里的外套已经晾干了。\n哦，我明白你的意思了。\n屋外的雾慢慢散开，路也看清楚了。",
    },
    {
        "title": "短句节奏",
        "batch": 2,
        "kind": "reading",
        "focus": "节奏和重音",
        "instruction": "第一遍平均速度读，第二遍把每句的关键词稍微加重，第三遍把句尾放松。不要改变字词，只练习节奏和重音位置。",
        "text": "现在出发。\n先听清楚。\n别急，慢一点。\n这一步很重要。\n我说完以后，你再回答。\n今天的重点，是把事情做好。",
    },
    {
        "title": "问句和陈述句",
        "batch": 2,
        "kind": "reading",
        "focus": "语调方向",
        "instruction": "带问号的句子句尾自然上扬，陈述句句尾自然落下。先按正常语气读，再稍微夸张一点对比，但不要像播音。",
        "text": "你已经准备好了吗？\n我们现在就出发吗？\n我已经把门关好了。\n明天上午十点见面。\n你想喝热水，还是喝茶？\n好的，那就这样决定。",
    },
    {
        "title": "从紧张到平静",
        "batch": 2,
        "kind": "reading",
        "focus": "情绪释放和收束",
        "instruction": "第一遍像刚遇到意外，语速略快；第二遍逐渐稳定；第三遍完全平静。不要提高到喊叫，用呼吸和停顿表现情绪变化。",
        "text": "等一下，我好像忘了什么。\n先别动，让我再检查一遍。\n好了，东西都在这里。\n不用担心，我们还有时间，慢慢来就好。",
    },
    {
        "title": "弱起和强起",
        "batch": 2,
        "kind": "singing",
        "focus": "起音力度和节拍",
        "instruction": "用自己的旋律清唱两遍。第一遍每行轻轻起音，第二遍每行前两个字更有支撑，后面保持放松。不要用喉咙硬顶起音。",
        "text": "轻轻说一声早安，\n让清晨慢慢醒来。\n我要把心打开，\n迎接今天的光彩。\n\n再向前走一步，\n把脚步唱得明亮。\n不管风吹向哪里，\n我都有自己的方向。",
    },
    {
        "title": "短音跳跃",
        "batch": 2,
        "kind": "singing",
        "focus": "短音、跳跃和音准反应",
        "instruction": "用短而清楚的音符唱，每行前半句可以轻快跳跃，后半句拉回稳定。没有固定音高，选择舒服旋律，不要追求很高或很快。",
        "text": "一闪一闪的灯，\n落在安静的城。\n一步一步地走，\n听见自己的歌。\n\n轻轻跳过石阶，\n绕过晚风和树影。\n停在熟悉的门前，\n把最后一句唱清。",
    },
    {
        "title": "同一元音换音高",
        "batch": 2,
        "kind": "singing",
        "focus": "换音高时的音色一致",
        "instruction": "先用“啊”在舒服的低、中、高三个音区各唱一次，再把短句唱两遍。换音高时保持口型和气息相近，不要为了高音挤压。",
        "text": "啊——啊——啊——\n我听见远方的回答。\n啊——啊——啊——\n把同一个声音放在不同的高度。",
    },
    {
        "title": "副歌收尾渐弱",
        "batch": 2,
        "kind": "singing",
        "focus": "副歌结束和渐弱",
        "instruction": "前四行像副歌一样打开，后四行逐渐收小。第二遍特别注意最后两个字仍然有音高和气息，不要突然掉声或挤喉。",
        "text": "我要把这首歌唱到天亮，\n让每一道光照在肩上。\n就算远方还有风浪，\n我也会守住最初的愿望。\n\n当最后一句慢慢落下，\n请听见我轻轻回答。\n故事不会停在这里，\n明天还会继续开花。",
    },
    {
        "title": "完整段落连贯演唱",
        "batch": 2,
        "kind": "singing",
        "focus": "连续演唱和整体稳定",
        "instruction": "把下面整段当成一小段歌曲连续唱两遍。第一遍保持舒服和稳定，第二遍加入自己的情绪变化。中间按换行换气，不要因为某个音不准而停下重来。",
        "text": "我从安静的街口出发，\n带着还没说完的话。\n风吹过肩头的时候，\n我知道答案正在路上。\n\n如果夜色遮住方向，\n就让歌声替我照亮。\n一步一步走到天亮，\n把自己的名字唱响。",
    },
    # Batch 3 (2026-11-17): the high register. Measured on the 36 takes recorded before: only 5 % of
    # the sung or spoken pitch lies above 272 Hz and 1 % above 391 Hz, while choruses of the songs
    # being covered sit at 400-520 Hz. A voice model cannot sound like its speaker on notes it never
    # heard from them, so these prompts ask for the highest notes that are still comfortable.
    {
        "title": "从舒服的音慢慢往高走",
        "batch": 3,
        "kind": "singing",
        "focus": "高音区（最要紧的一组）",
        "instruction": "用“啊”唱。先唱一个最舒服的音，拖三秒；然后每次高一点，再拖三秒，一直唱到“再高就费劲”的地方停下，然后原路走回来。整个过程做两遍。声音可以变细、变轻，不要喊。",
        "text": "啊——（舒服的音）\n啊——（高一点）\n啊——（再高一点）\n啊——（再高一点）\n啊——（你能轻松唱到的最高处）\n\n然后一个一个往回走，回到开始的音。",
    },
    {
        "title": "在高处唱一句话",
        "batch": 3,
        "kind": "singing",
        "focus": "高音区的咬字",
        "instruction": "把下面每一行都用偏高的音唱出来，像歌曲的副歌那样，比你平时哼歌的音高高出一截。每个字都要唱清楚。唱两遍，第二遍比第一遍再高一点点。唱不上去就降一点，别硬撑。",
        "text": "我要飞到云的上面，\n看一看更远的明天。\n就算风把我吹得很远，\n我也不会忘记这一天。",
    },
    {
        "title": "假声（很轻很高的声音）",
        "batch": 3,
        "kind": "singing",
        "focus": "假声",
        "instruction": "用假声唱：就是学猫头鹰“呜——”的那种又轻又高的声音，不用力。先用“呜”唱三个长音，再用同样的声音把短句唱两遍。假声是很多歌高音部分的唱法，录下来以后模型才学得会。",
        "text": "呜——\n呜——\n呜——\n\n月亮慢慢升起来，\n照着我也照着你。\n轻轻唱给夜晚听，\n不用很大的声音。",
    },
    {
        "title": "低处唱一句，高处再唱一句",
        "batch": 3,
        "kind": "singing",
        "focus": "同一句话的不同音高",
        "instruction": "每一行唱三次：先用低而舒服的音，再用中间的音，最后用偏高的音。字和节奏不变，只换音高。这样模型能学到同一个字在你不同音高上的样子。",
        "text": "天亮了，我出发了。\n路很长，我不着急。\n你听见了吗，我在唱歌。\n再见了，明天再来。",
    },
    {
        "title": "副歌：从中间一直唱到最高",
        "batch": 3,
        "kind": "singing",
        "focus": "高音区的连贯演唱",
        "instruction": "把下面当成一段副歌，用自己的旋律连着唱两遍。前两行在中间音高，后面一行比一行高，最后一行是整段的最高处，句尾的字拖长两三秒。高音唱得细一点没关系，重要的是别断。",
        "text": "我走过很多的路，\n也看过很多的灯。\n可是只有这一刻，\n我想把歌唱给你听。\n\n唱到天空都亮起来，\n唱到风也停下来，\n唱到你也跟着我——\n一起大声唱出来——",
    },
    {
        "title": "用偏高的声音说话",
        "batch": 3,
        "kind": "reading",
        "focus": "高音区的说话声",
        "instruction": "用比平时高一截的声音读，像隔着一条马路喊朋友，又像逗小孩子说话。不用喊，声音放亮、放高就行。读两遍。",
        "text": "喂——你在哪儿呀？我在这边！\n快过来看，这里有好多花！\n哇，今天的天气真好啊！\n等等我，我马上就来！\n哎呀，原来你在这儿，我找了你好久！",
    },
]


def first_new_prompt_index() -> int:
    """Return the first supplemental prompt; the original 12 remain available."""
    return next((index for index, prompt in enumerate(PROMPTS) if prompt.get("focus")), len(PROMPTS))


def prompt_batch(index) -> int:
    position = _index(index)
    return int(PROMPTS[position].get("batch", 1))


def supplemental_batch_indices(batch: int) -> list[int]:
    return [index for index, prompt in enumerate(PROMPTS)
            if prompt.get("focus") and int(prompt.get("batch", 1)) == int(batch)]


BATCH_NAMES = {1: "补录第 1 组", 2: "补录第 2 组", 3: "高音专项"}


def prompt_label(index) -> str:
    position = _index(index)
    prompt = PROMPTS[position]
    if not prompt.get("focus"):
        return f"基础稿 第 {position + 1} 段，共 12 段"
    batch = prompt_batch(position)
    group = supplemental_batch_indices(batch)
    return f"{BATCH_NAMES.get(batch, f'补录第 {batch} 组')} 第 {group.index(position) + 1} 段，共 {len(group)} 段"


def next_prompt_index(index) -> int:
    """Advance legacy prompts into supplements, then loop only within the same batch."""
    position = _index(index)
    first_new = first_new_prompt_index()
    if first_new >= len(PROMPTS):
        return (position + 1) % len(PROMPTS)
    if position >= first_new:
        group = supplemental_batch_indices(prompt_batch(position))
        next_positions = [candidate for candidate in group if candidate > position]
        if next_positions:
            return next_positions[0]
        next_position = position + 1
        if next_position < len(PROMPTS) and next_position >= first_new:
            if prompt_batch(next_position) != prompt_batch(position):
                return next_position
        return group[0]
    return position + 1 if position + 1 < first_new else first_new


def _index(index) -> int:
    try:
        return int(index) % len(PROMPTS)
    except (TypeError, ValueError, OverflowError):
        return 0


def prompt_kind(index) -> str:
    position = _index(index)
    kind = PROMPTS[position].get("kind")
    return kind or ("singing" if 8 <= position < 12 else "reading")


def prompt_markdown(index) -> str:
    """The card shown above the microphone: what to do, how, and the words."""
    position = _index(index)
    prompt = PROMPTS[position]
    kind = prompt_kind(position)
    task = "清唱（不放伴奏，自己哼旋律）" if kind == "singing" else "朗读"
    lines = "".join(f"<p>{html.escape(line)}</p>" if line else '<p class="gap"></p>' for line in prompt["text"].splitlines())
    return (f'<section class="sheet sheet-{kind}">'
            f'<p class="sheet-where">{html.escape(prompt_label(position))}</p>'
            f'<h3>{html.escape(prompt["title"])}</h3>'
            f'<p class="sheet-task"><b>{task}</b></p>'
            f'<p class="sheet-how">{html.escape(prompt["instruction"])}</p>'
            f'<div class="sheet-words">{lines}</div>'
            '</section>')


def recording_stats() -> dict:
    """Count actual, decodable audio; a duration is never inferred from filenames."""
    files = []
    seconds = 0.0
    reading_seconds = 0.0
    singing_seconds = 0.0
    metadata_by_name = {}
    if RECORD_INFO.exists():
        for metadata_path in RECORD_INFO.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                name = Path(metadata["saved_filename"]).name
                if name == metadata["saved_filename"]:
                    metadata_by_name[name] = metadata
            except (OSError, ValueError, TypeError, KeyError):
                continue
    if DATASET.exists():
        for path in sorted(DATASET.iterdir()):
            if not path.is_file() or path.suffix.lower() not in _AUDIO_EXTENSIONS:
                continue
            try:
                info = sf.info(str(path))
                duration = info.frames / info.samplerate if info.samplerate else 0.0
            except (OSError, RuntimeError, ValueError):
                continue
            if duration > 0 and math.isfinite(duration):
                files.append(str(path))
                seconds += duration
                metadata = metadata_by_name.get(path.name, {})
                prompt_index = metadata.get("prompt_index")
                # Older metadata predates ``kind``; preserve the original 8-11 singing split.
                kind = metadata.get("prompt_kind")
                if kind is None and isinstance(prompt_index, int):
                    kind = "singing" if 8 <= prompt_index < 12 else "reading"
                if kind == "singing":
                    singing_seconds += duration
                else:
                    reading_seconds += duration
    return {"files": files, "count": len(files), "seconds": seconds,
            "reading_seconds": reading_seconds, "singing_seconds": singing_seconds}


def recorded_prompts() -> set[int]:
    """Prompt numbers that have at least one saved take."""
    recorded = set()
    if RECORD_INFO.exists():
        for path in RECORD_INFO.glob("*.json"):
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
                position = int(info["prompt_index"])
                if 0 <= position < len(PROMPTS) and (DATASET / Path(info["saved_filename"]).name).is_file():
                    recorded.add(position)
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return recorded


def high_note_progress() -> tuple[int, int]:
    """(recorded, total) for the high-note prompts (batch 3)."""
    group = supplemental_batch_indices(3)
    done = recorded_prompts()
    return sum(1 for position in group if position in done), len(group)


def _duration(seconds: float) -> str:
    rounded = int(round(seconds))
    return f"{rounded // 60} 分 {rounded % 60:02d} 秒"


def summary() -> str:
    stats = recording_stats()
    text = (f"已保存 {stats['count']} 段 · 总计 {_duration(stats['seconds'])}"
            f"（朗读 {_duration(stats['reading_seconds'])} / 清唱 {_duration(stats['singing_seconds'])}）。")
    if stats["seconds"] < 300:
        text += f" 距最低训练时长还差 {_duration(300 - stats['seconds'])}。"
    elif stats["seconds"] < 600:
        text += " 已达到最低训练时长；可优先补充舒服音区的清唱。"
    else:
        text += " 已超过 10 分钟；还可补充不同语气和音区的清唱。"
    return text


def initial_index() -> int:
    """Resume unfinished basics for newcomers; keep existing supplement users there."""
    first_new = first_new_prompt_index()
    recorded = set()
    latest = None
    if RECORD_INFO.exists():
        for path in RECORD_INFO.glob("*.json"):
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
                position = int(info["prompt_index"])
                if not 0 <= position < len(PROMPTS):
                    continue
                recorded.add(position)
                stamp = path.stat().st_mtime_ns
            except (OSError, ValueError, TypeError, KeyError):
                continue
            if latest is None or stamp > latest[0]:
                latest = (stamp, position)
    # Fresh installs should start with the foundations, and refreshing partway
    # through them must not jump to supplemental material. Once a user has
    # recorded any supplemental prompt, retain the existing supplement-first
    # behavior even if they skipped some basics.
    start = first_new if any(position >= first_new for position in recorded) else 0
    for position in range(start, len(PROMPTS)):
        if position not in recorded:
            return position
    if latest is not None:
        return next_prompt_index(latest[1])
    return first_new if first_new < len(PROMPTS) else 0


def _inspect_audio(path: Path) -> dict:
    sample_count = 0
    square_sum = 0.0
    peak = 0.0
    clipped = 0
    gated = 0      # samples inside runs of exact digital zero lasting at least 10 ms
    zero_run = 0
    with sf.SoundFile(str(path)) as recording:
        if not recording.frames or not recording.samplerate:
            raise ValueError("这段录音是空的，请点击麦克风重新录制。")
        info = {
            "seconds": recording.frames / recording.samplerate,
            "sample_rate": recording.samplerate,
            "channels": recording.channels,
            "format": recording.format,
        }
        for block in recording.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                raise ValueError("录音中有无效音频数据，请重新录制这一段。")
            if block.size:
                peak = max(peak, float(np.abs(block).max()))
                square_sum += float(np.sum(np.square(block, dtype=np.float64)))
                clipped += int(np.count_nonzero(np.abs(block) >= 0.999))
                sample_count += block.size
                # A real microphone never outputs exact zeros for 10 ms; a noise
                # gate (browser or driver "noise suppression") does.
                minimum = max(1, recording.samplerate // 100)
                silent = np.concatenate([[False], ~block.any(axis=1), [False]])
                edges = np.flatnonzero(silent[1:] != silent[:-1])
                for start, end in zip(edges[::2].tolist(), edges[1::2].tolist()):
                    length = end - start + (zero_run if start == 0 else 0)
                    if start == 0 and zero_run >= minimum:
                        gated += end - start
                    elif length >= minimum:
                        gated += length
                    zero_run = length if end == len(block) else 0
                if not silent[-2]:
                    zero_run = 0
    if not sample_count or info["seconds"] < 0.25:
        raise ValueError("这段录音太短，可能还没录到声音。请重新录制。")
    rms = math.sqrt(square_sum / sample_count)
    if peak < 0.0003 and rms < 0.00001:
        raise ValueError("几乎没有录到声音，这段没有保存。请检查浏览器麦克风权限和输入设备，再试一次。")
    frames = sample_count / max(info["channels"], 1)
    info.update({"peak": peak, "rms": rms, "clipped_ratio": clipped / sample_count,
                 "gated_ratio": gated / frames if frames else 0.0})
    return info


def _source_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _duplicate(source_sha256: str) -> str | None:
    if not RECORD_INFO.exists():
        return None
    for path in RECORD_INFO.glob("*.json"):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            saved_name = info["saved_filename"]
            if (info.get("source_sha256") == source_sha256
                    and Path(saved_name).name == saved_name
                    and (DATASET / saved_name).is_file()):
                return saved_name
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def save_recording(path, index) -> tuple:
    """Save one take, then show the next prompt. Existing recordings stay intact."""
    position = _index(index)
    if not path:
        raise ValueError("请先点击麦克风录音，读完后点击停止，再保存这一段。")
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("没有找到录音文件，请重新录制这一段。")
    try:
        audio_info = _inspect_audio(source)
    except (OSError, RuntimeError) as exc:
        raise ValueError("这段音频无法读取，请重新录音，或上传 WAV、FLAC 音频。") from exc
    source_sha256 = _source_hash(source)
    with _SAVE_LOCK:
        DATASET.mkdir(parents=True, exist_ok=True)
        RECORD_INFO.mkdir(parents=True, exist_ok=True)
        duplicate_name = _duplicate(source_sha256)
        if duplicate_name:
            next_index = next_prompt_index(position)
            return (str(DATASET / duplicate_name), next_index, prompt_markdown(next_index), summary(),
                    "这段录音已经保存过，没有重复添加。可以继续录下一段。")
        name = f"voice_{datetime.now():%Y%m%d_%H%M%S}_{position + 1:02d}_{uuid.uuid4().hex[:8]}.wav"
        destination = DATASET / name
        temporary = DATASET / f".{name}.part"
        try:
            if audio_info["format"] in {"WAV", "WAVEX", "RF64"}:
                # Preserve microphone WAV samples, levels, channels and sample rate.
                shutil.copyfile(source, temporary)
            else:
                # Other supported input formats become lossless PCM; no gain or filtering.
                with sf.SoundFile(str(source)) as recording, sf.SoundFile(
                    str(temporary), mode="x", samplerate=recording.samplerate,
                    channels=recording.channels, format="WAV", subtype="FLOAT",
                ) as output:
                    for block in recording.blocks(blocksize=65536, dtype="float32", always_2d=True):
                        output.write(block)
            temporary.replace(destination)
            metadata = {
                "saved_filename": name, "source_sha256": source_sha256,
                "prompt_index": position, "prompt_title": PROMPTS[position]["title"],
                "prompt_batch": prompt_batch(position),
                "prompt_kind": PROMPTS[position].get("kind", "singing" if position >= 8 and position < 12 else "reading"),
                "saved_at": datetime.now().astimezone().isoformat(), **audio_info,
            }
            (RECORD_INFO / f"{name}.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        finally:
            temporary.unlink(missing_ok=True)
    next_index = next_prompt_index(position)
    feedback = f"已保存，这段长 {_duration(audio_info['seconds'])}。"
    if audio_info["clipped_ratio"] > 0.001 or audio_info["peak"] > 1.0:
        feedback += " 这段有点爆音（声音太大被削顶了）。下一段离麦克风远一点，或把麦克风音量调小；介意的话可以重录这段。"
    elif audio_info["rms"] < 0.003:
        feedback += " 这段声音偏小。下一段可以离麦克风近一点，不用刻意喊。"
    if audio_info.get("gated_ratio", 0) > 0.02:
        feedback += (f" 注意：这段有 {audio_info['gated_ratio'] * 100:.0f}% 是“纯数字静音”，说明声音在进入网页前"
                     "被降噪处理过（字与字之间的气声、尾音可能被切掉，不利于练出清楚的咬字）。请刷新本页后重录；"
                     "如果还出现，请在 Windows 声音设置或麦克风、声卡的软件里关掉“噪音抑制、语音增强、AI 降噪”。")
    if position == len(PROMPTS) - 1:
        feedback += " 这一组已经录完了，接下来会回到这一组的第一段，想多录可以继续。"
    # Keep the saved take in the player so the user can listen again after saving.
    return str(destination), next_index, prompt_markdown(next_index), summary(), feedback


def clear_current(index) -> tuple:
    return None, prompt_markdown(index), "刚才没保存的录音已经清掉，可以重录这一段。保存过的录音不受影响。"
