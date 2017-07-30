--2017.7.20 记录 --:
问题：为什么MC的off-policy更新过程有latest time时间点，而TD的off-policy过程没有？
答案：
--2017.7.30 记录--:
问题：不断修改MC的on-policy代码，总是看不到效果，有几个原因？
答案：有如下几个原因：1、训练过程的return有时比验证过程随机，因为训练过程要随机选择action，验证过程可以始终用greedy action。但是如果训练的正确，无论训练还是验证过程，都应看到return明显提升。
2、代码细节问题。这次最终找到的问题是：在选择action时使用了np.where（）函数来获取index，但是使用方式不对。关键代码是这一句：sa_pair = (c_ep[i],np.where((np.array(self.actions)==c_ep[i+1]).all(1))[0][0])
3、在从一个list中选择最大值时，要时刻考虑到最大值也许会有多个，需要从这多个中随机选取，不然就固定选择首次出现的那个了。关键代码是这一句：best_action = self.actions[np.random.choice(np.where(self.Q[s] == np.amax(self.Q[s]))[0])]







