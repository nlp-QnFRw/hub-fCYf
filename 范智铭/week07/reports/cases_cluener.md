# 定性对比样例（cluener）

## bilstm 对 / roberta 错 · `cluener-test-000031`

```text
贝蒂斯同样也是一支打法开放的球队，双方本场将会大打攻势足球，支持希洪本轮依然没有平局，
```

金标：name「贝蒂斯」、organization「希洪」
roberta 漏/错：name「贝蒂斯」
roberta 多标：organization「贝蒂斯」、organization「希洪」

## bilstm 对 / roberta 错 · `cluener-test-000112`

```text
战争世界》的剧情背景与人物，玩家将扮演身为CIA特勤人员的男主角Mason，
```

金标：game「战争世界》」、name「Mason」
roberta 漏/错：game「战争世界》」
roberta 多标：name「Mason」、government「CIA」、movie「战争世界》」、position「特勤人员」

## bilstm 对 / roberta 错 · `cluener-test-000135`

```text
高龙岛是柬埔寨南部城市西哈努克的一个离岛，未经大规模商业开发，高龙岛总面积约78平方公里，
```

金标：scene「高龙岛」、address「柬埔寨南部城市西哈努克」、scene「高龙岛」
roberta 漏/错：address「柬埔寨南部城市西哈努克」
roberta 多标：address「西哈努克」、scene「高龙岛」、address「柬埔寨南部城市」、scene「高龙岛」

## bilstm 对 / llm_zero 错 · `cluener-test-000050`

```text
当前平手盘更为顾忌赫尔城的出色表现，预计受注客队会偏多，此盘不太容易打平；平手盘赔率下，威廉、立博、
```

金标：organization「赫尔城」、company「威廉」、company「立博」
llm_zero 漏/错：organization「赫尔城」、company「威廉」、company「立博」

## bilstm 对 / llm_zero 错 · `cluener-test-000091`

```text
14拉齐奥vs国际米兰
```

金标：organization「拉齐奥」、organization「国际米兰」
llm_zero 漏/错：organization「拉齐奥」、organization「国际米兰」

## bilstm 对 / llm_zero 错 · `cluener-test-000103`

```text
调整前的威廉同赔率下，英超主队4胜3平1负；提高赔付率时，威廉同时抬高了平赔和客胜赔，同赔率下，
```

金标：company「威廉」、organization「英超」、company「威廉」
llm_zero 漏/错：company「威廉」、organization「英超」、company「威廉」

## bilstm 对 / llm_few 错 · `cluener-test-000038`

```text
游戏制作商为了掩盖一些程序上的错误，就立刻发布了《毁灭公爵2》。当然，修改人贾斯汀·
```

金标：game「《毁灭公爵2》」、name「贾斯汀·」
llm_few 漏/错：game「《毁灭公爵2》」
llm_few 多标：name「贾斯汀·」、game「毁灭公爵2」

## bilstm 对 / llm_few 错 · `cluener-test-000103`

```text
调整前的威廉同赔率下，英超主队4胜3平1负；提高赔付率时，威廉同时抬高了平赔和客胜赔，同赔率下，
```

金标：company「威廉」、organization「英超」、company「威廉」
llm_few 漏/错：company「威廉」
llm_few 多标：organization「英超」、company「威廉」

## bilstm 对 / llm_few 错 · `cluener-test-000138`

```text
印度尼西亚海军一架海上侦察机7日在东加里曼丹省执行日常巡逻任务时与地面失去联系，机上有9名机组人员。
```

金标：government「印度尼西亚海军」、address「东加里曼丹省」
llm_few 漏/错：government「印度尼西亚海军」
llm_few 多标：address「东加里曼丹省」、organization「印度尼西亚海军」

## bert 对 / bilstm 错 · `cluener-test-000001`

```text
绵阳江油李白故里、佛爷洞、九皇山平武报恩寺白马藏寨白马王朗国家级自然保护区杜鹃山
```

金标：scene「绵阳江油李白故里」、scene「佛爷洞」、scene「九皇山」、scene「平武报恩寺」、scene「白马藏寨」、scene「白马王朗国家级自然保护区」、scene「杜鹃山」
bilstm 漏/错：scene「绵阳江油李白故里」、scene「九皇山」、scene「平武报恩寺」、scene「白马藏寨」、scene「白马王朗国家级自然保护区」、scene「杜鹃山」
bilstm 多标：scene「李白故里」、scene「佛爷洞」、address「绵阳江油」、scene「九皇山平武报恩寺白马藏寨白马王朗国家级自然保护区杜鹃山」

## bert 对 / bilstm 错 · `cluener-test-000006`

```text
2004年郑芳芳在景湖路办起了第一家福彩投注站，第一年站点月销量就在6—7万元之间。
```

金标：name「郑芳芳」、address「景湖路」
bilstm 漏/错：name「郑芳芳」、address「景湖路」
bilstm 多标：address「景湖路办」

## bert 对 / bilstm 错 · `cluener-test-000013`

```text
所有影片的共同制作和发行权。斯皮尔伯格将继续担任《变形金刚》系列的制片人，
```

金标：name「斯皮尔伯格」、movie「《变形金刚》」、position「制片人」
bilstm 漏/错：position「制片人」
bilstm 多标：name「斯皮尔伯格」、movie「《变形金刚》」

## bert 对 / roberta 错 · `cluener-test-000015`

```text
5月26日，刚上市的德国游戏杂志《Gamestar》刊载了《辐射：新拉斯维加斯》
```

金标：book「《Gamestar》」、book「《辐射：新拉斯维加斯》」
roberta 漏/错：book「《辐射：新拉斯维加斯》」
roberta 多标：book「《Gamestar》」、game「《辐射：新拉斯维加斯》」

## bert 对 / roberta 错 · `cluener-test-000084`

```text
谢林曦：新浪乐居数据资源管理部总监、《购房指南针》主编
```

金标：name「谢林曦」、company「新浪乐居数据资源管理部」、position「总监」、book「《购房指南针》」、position「主编」
roberta 漏/错：company「新浪乐居数据资源管理部」
roberta 多标：book「《购房指南针》」、company「据资源管」、company「新浪乐居」、position「总监」、position「主编」

## bert 对 / roberta 错 · `cluener-test-000112`

```text
战争世界》的剧情背景与人物，玩家将扮演身为CIA特勤人员的男主角Mason，
```

金标：game「战争世界》」、name「Mason」
roberta 漏/错：game「战争世界》」
roberta 多标：name「Mason」、government「CIA」、movie「战争世界》」、position「特勤人员」

## bert 对 / llm_zero 错 · `cluener-test-000022`

```text
主持人：下面，我们想请陈先生来谈一下，您所创办的《号外杂志》
```

金标：position「主持人」、book「《号外杂志》」
llm_zero 漏/错：position「主持人」、book「《号外杂志》」
llm_zero 多标：book「号外杂志」、name「陈先生」

## bert 对 / llm_zero 错 · `cluener-test-000034`

```text
曾梵志：这样的话好多人问过，很多收藏家问，批评家问的倒不是很多，收藏家问，可能是出于购买“
```

金标：name「曾梵志」、position「收藏家」、position「批评家」、position「收藏家」
llm_zero 漏/错：position「收藏家」、position「批评家」、position「收藏家」
llm_zero 多标：name「曾梵志」

## bert 对 / llm_zero 错 · `cluener-test-000049`

```text
法国画家塞尚、莫奈等大师的作品在内的一批油画，情况都已十分危急。
```

金标：position「画家」、name「塞尚」、name「莫奈」、position「大师」
llm_zero 漏/错：position「画家」、position「大师」
llm_zero 多标：name「塞尚」、name「莫奈」、address「法国」

## bert 对 / llm_few 错 · `cluener-test-000022`

```text
主持人：下面，我们想请陈先生来谈一下，您所创办的《号外杂志》
```

金标：position「主持人」、book「《号外杂志》」
llm_few 漏/错：book「《号外杂志》」
llm_few 多标：book「号外杂志」、position「主持人」、name「陈先生」

## bert 对 / llm_few 错 · `cluener-test-000034`

```text
曾梵志：这样的话好多人问过，很多收藏家问，批评家问的倒不是很多，收藏家问，可能是出于购买“
```

金标：name「曾梵志」、position「收藏家」、position「批评家」、position「收藏家」
llm_few 漏/错：position「收藏家」、position「批评家」、position「收藏家」
llm_few 多标：name「曾梵志」

## bert 对 / llm_few 错 · `cluener-test-000038`

```text
游戏制作商为了掩盖一些程序上的错误，就立刻发布了《毁灭公爵2》。当然，修改人贾斯汀·
```

金标：game「《毁灭公爵2》」、name「贾斯汀·」
llm_few 漏/错：game「《毁灭公爵2》」
llm_few 多标：name「贾斯汀·」、game「毁灭公爵2」

## llm_zero 对 / roberta 错 · `cluener-test-000272`

```text
从目前来看，花旗、汇丰都受到了次贷的重创，但其在国内保持了较低的存贷比水平，因此花旗中国、
```

金标：company「花旗」、company「汇丰」、company「花旗中国」
roberta 漏/错：company「花旗中国」
roberta 多标：company「国」、company「汇丰」、company「花旗」、company「花旗」

## llm_zero 对 / roberta 错 · `cluener-test-000403`

```text
当期浙江的另1注双色球头奖在杭州市凤起路第33010122号站点中出，
```

金标：address「浙江」、address「杭州市凤起路第33010122号站点」
roberta 漏/错：address「杭州市凤起路第33010122号站点」
roberta 多标：address「浙江」、address「杭州市凤起路第33010122号站」

## llm_zero 对 / roberta 错 · `cluener-test-000431`

```text
正是他14部武侠小说书名的第一个字。还一部不在其中的，便是《越女剑》。资料图金庸作品。唐鸭鸭
```

金标：book「《越女剑》」、name「金庸」、name「唐鸭鸭」
roberta 漏/错：name「唐鸭鸭」
roberta 多标：book「《越女剑》」、name「金庸」

## llm_few 对 / roberta 错 · `cluener-test-000157`

```text
詹俊足彩推荐：蓝军红军均要防平枪手热刺作稳胆
```

金标：name「詹俊」、organization「蓝军」、organization「红军」、organization「热刺」
roberta 漏/错：organization「蓝军」、organization「红军」
roberta 多标：organization「足」、name「詹俊」、organization「热刺」、organization「枪手」

## llm_few 对 / roberta 错 · `cluener-test-000261`

```text
俄罗斯驻委内瑞拉大使馆消息，俄罗斯现阶段正在委境内修建多座兵工厂，准备生产ak－
```

金标：government「俄罗斯驻委内瑞拉大使馆」、address「俄罗斯」
roberta 漏/错：address「俄罗斯」
roberta 多标：government「俄罗斯驻委内瑞拉大使馆」

## llm_few 对 / roberta 错 · `cluener-test-000272`

```text
从目前来看，花旗、汇丰都受到了次贷的重创，但其在国内保持了较低的存贷比水平，因此花旗中国、
```

金标：company「花旗」、company「汇丰」、company「花旗中国」
roberta 漏/错：company「花旗中国」
roberta 多标：company「国」、company「汇丰」、company「花旗」、company「花旗」

## llm_few 对 / llm_zero 错 · `cluener-test-000050`

```text
当前平手盘更为顾忌赫尔城的出色表现，预计受注客队会偏多，此盘不太容易打平；平手盘赔率下，威廉、立博、
```

金标：organization「赫尔城」、company「威廉」、company「立博」
llm_zero 漏/错：organization「赫尔城」、company「威廉」、company「立博」

## llm_few 对 / llm_zero 错 · `cluener-test-000091`

```text
14拉齐奥vs国际米兰
```

金标：organization「拉齐奥」、organization「国际米兰」
llm_zero 漏/错：organization「拉齐奥」、organization「国际米兰」

## llm_few 对 / llm_zero 错 · `cluener-test-000153`

```text
星际争霸项目最具影响力的解说员，教练员——大师将倾情加盟此次春晚。
```

金标：game「星际争霸」、position「解说员」、position「教练员」
llm_zero 漏/错：position「解说员」、position「教练员」
llm_zero 多标：game「星际争霸」、name「大师」

