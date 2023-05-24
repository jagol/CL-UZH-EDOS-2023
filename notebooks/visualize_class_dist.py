import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


vectors = (
    'sexist',
    'not sexist',
)
y_pos = np.arange(len(vectors))
vector_dist = [
    24.67, 
    75.53, 
]

score_series = pd.Series(vector_dist)

# Plot the figure.
plt.figure(figsize=(12, 8))
fig = score_series.plot(kind='bar')
fig.set_xticklabels(vectors)
fig.bar_label(fig.containers[0], label_type='edge')

ax = plt.gca()
#hide x-axis
ax.get_xaxis().set_visible(False)
#hide y-axis 
ax.get_yaxis().set_visible(False)



plt.savefig('/srv/scratch0/jgoldz/CL-UZH-EDOS-2023/output/class_dist/bin_dist.jpg')




vectors = (
    '1.1 Threats of harm', 
    '1.2 Incitement and encouragement of harm',
    '2.1 descriptive attack',
    '2.2 aggressive and emotive attacks',
    '2.3 Dehumanising attacks and overt sexual objectification',
    '3.1 casual use of gendered slurs, profanities, and insult',
    '3.2 immutable gender differences and gender stereotypes',
    '3.3 backhanded gendered compliments',
    '3.4 condescending explanations or unwelcome advice',
    '4.1 supporting mistreatment of individual women',
    '4.2 supporting systemic discrimination against women as a group'
)
y_pos = np.arange(len(vectors))
vector_dist = [
    1.46, 
    7.32, 
    21.22,
    19.55,
    5.72,
    18.97,
    12.40,
    2.01,
    1.43,
    2.25,
    7.66
]

plt.bar(y_pos, vector_dist, align='center', alpha=0.5)
plt.xticks(y_pos, vectors)
# plt.ylabel('Usage')
# plt.title('Programming language usage')

plt.savefig('/srv/scratch0/jgoldz/CL-UZH-EDOS-2023/output/class_dist/vector_dist.jpg')
