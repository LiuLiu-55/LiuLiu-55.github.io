---
layout:     post
title:      个人博客搭建(基于github pages和jekyll)
subtitle:   
date:       2024-03-12
author:     LiuLiu
header-img: img/post-bg-mountain2.jpg
catalog: true
categories: 博客
tags:
    - Blog
---

# 前言

从很久开始就一直想自己弄一个博客，可以记录自己的学习成果，还可以装一下逼。但是因为自己太懒，一直拖到了现在。最近发现如果没有记录的话，学过的知识真的很容易忘掉，万花从中过，片叶不沾身。脑子空空，痴痴傻傻。虽然也有把笔记写在Obsidian上，但缺乏一个监管机制，所以笔记也很乱哈哈哈，写的乱七八糟。而且马上就要就业了，有一个博客也许可以成为一个加分项？（我也不知道，先做吧）所以，痛定思痛，我终于采取了行动，开始捣鼓自己的博客。自己的笔记虽然有点幼稚，不如网上那些大神，但有总比没有好吧哈哈哈哈。

# 准备

## 下载Ruby、gem、jekyll、Bundler

**安装Ruby**

官网：https://www.ruby-lang.org/zh_cn/documentation/installation/#other-systems

利用`apt-get`命令安装

```sudo apt-get install ruby-full```

发现安装的版本很低，为`2.7.9`

版本太低，无法安装jekyll，所以改为源码安装ruby



首先[官网](https://www.ruby-lang.org/zh_cn/downloads/)下载最新版本`ruby3.3.0`

解压后进入，按照官方文档进行安装

```bash
tar -xvzf ruby-3.3.0.tar.gz 
cd ruby-3.3.0
./configure
make
sudo make install
```

报错

```bash
warning: It seems your ruby installation is missing psych (for YAML output).
```

根据[stackoverflow](https://stackoverflow.com/questions/15738883/please-install-libyaml-and-reinstall-your-ruby)的回答，需要额外安装libyaml

从https://pyyaml.org/download/libyaml/下载yaml安装包进行安装

```bash
tar -xvzf yaml-0.2.5.tar.gz
cd yaml-0.2.5
./configure
make
sudo make install
```

安装完`yaml`后，重新编译`ruby`

```bash
cd ruby-3.3.0
make clean
./configure
make
sudo make install
```

注意，需要用`make clean`后才能重新安装成功

***

**安装gem**

按照[jekyll 官网](https://jekyllrb.com/)的步骤可以直接使用`sudo apt-get install ruby-full build-essential zlib1g-dev`安装ruby和gem，但版本太低，所以需要对gem进行升级。按照[RubyGems官网](https://rubygems.org/pages/download)更新gem

```bash
gem update --system
```

***

**安装jekyll、Bundler**

```bash
gem install jekyll bundler
```



# 配置

首先fork一个[repostory](https://github.com/Xiaokeai18/xiaokeai18.github.io)

之后更改自己的配置，在`_config.yml`中进行更改

```yaml
# Site settings
title: LIULIU's Blog
SEOTitle: 刘柳的博客 | LIULIU's Blog
header-img: img/post-bg-desk.jpg
email: liuliucn@outlook.com
description: "这是刘柳的博客"
keyword: "LIU,LIU,刘柳，刘柳的博客"
url: "http://LiuLiu-55.github.io"          # your host, for absolute URL
baseurl: ""      # for example, '/blog' if your blog hosted on 'host/blog'
github_repo: "https://github.com/LiuLiu-55/LiuLiu-55.github.io.git" # you code repository

# Sidebar settings
sidebar: true                           # whether or not using Sidebar.
sidebar-about-description: "这是刘柳的博客~"
sidebar-avatar: /img/avatar.jpeg
```

首先就是要把首页的描述改成自己的

然后`url`和`repo`也要修改成自己的



gitalk也需要改成自己的

```yml
# Gitalk
gitalk:
  enable: true    #是否开启Gitalk评论
  clientID: ad0efad6219ca8022434                             #生成的clientID
  clientSecret: bb81c32a8144dfdba248a5fbf144cf444faf4732    #生成的clientSecret
  repo: LiuLiu-55.github.io    #仓库名称
  owner: LiuLiu-55    #github用户名
  admin: LiuLiu-55
  distractionFreeMode: true #是否启用类似FB的阴影遮罩
```

其中`clientID`与`clientSecret`需要由Github的`OAuth Apps`生成

具体的生成步骤参考xxx

https://xy-bz.github.io/2020/02/18/gitalk%E6%8F%92%E4%BB%B6%E6%B7%BB%E5%8A%A0/

https://calpa.me/blog/gitalk-error-validation-failed-442-solution/

https://blog.csdn.net/Mart1nn/article/details/87478971

https://blog.csdn.net/qq_38463737/article/details/120288329

https://blog.csdn.net/qq_38463737/article/details/120288329

https://wangpei.ink/2017/12/19/%E4%B8%BA%E5%8D%9A%E5%AE%A2%E6%B7%BB%E5%8A%A0-Gitalk-%E8%AF%84%E8%AE%BA%E6%8F%92%E4%BB%B6/

https://zhuanlan.zhihu.com/p/341543249



如果需要更改jekyll在本地调试，需要将`_config.yml`中的

```yml
gems: [jekyll-paginate]
```

更改为

```yml
plugins: [jekyll-paginate]
```

之后在博客根目录下输入

```shell
jekyll s
```

即可在本地查看博客。由于本博客是由jekyll+github创建的，所以不需要用到bundler，如果想要在本地建立一个博客。可以依照[教程](https://zhuanlan.zhihu.com/p/570464581)使用。

# 代办
- [ ] Gitalk
- [ ] Obsidian转jekyll
  - [ ] [PicGo](https://github.com/Molunerfinn/PicGo)
  - [ ] [Auto upload](https://github.com/renmu123/obsidian-image-auto-upload-plugin)