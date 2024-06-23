---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Markdown, HTML, TeX and LaTeX Guide
Refer the links for details: Basic, Extended, Hacks, Tools Etc.
- https://en.wikipedia.org/wiki/Markdown
- https://www.markdownguide.org/cheat-sheet
- https://www.ibm.com/docs/en/db2-event-store/2.0.0?topic=notebooks-markdown-jupyter-cheatsheet
- https://www.kaggle.com/code/marcovasquez/useful-html-for-jupyter-notebook

TeX and LaTeX
- https://en.wikibooks.org/wiki/LaTeX
- https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions


## Basics
1. Headings: To create a heading, add number signs (#) in front of a word or phrase. The number of number signs you use should correspond to the heading level.
2. Paragraphs: To create paragraphs, use a blank line to separate one or more lines of text.
3. Line Breaks: To create a line break or new line, end a line with two or more spaces, and then type return.
4. Bold: To bold text, add two asterisks before and after a word or phrase. To bold the middle of a word for emphasis, add two asterisks without spaces around the letters.
5. Italics: To italicize text, add one asterisk before and after a word or phrase. To italicize the middle of a word for emphasis, add one asterisk without spaces around the letters.
6. Bold and Italic: To emphasize text with bold and italics at the same time, add three asterisks before and after a word or phrase. To bold and italicize the middle of a word for emphasis, add three asterisks without spaces around the letters.
7. Blockquotes: To create a blockquote, add a > in front of a paragraph.
8. Ordered Lists: To create an ordered list, add line items with numbers followed by periods. The numbers don’t have to be in numerical order, but the list should start with the number one.
9. Unordered Lists: To create an unordered list, add dashes (-), asterisks (*), or plus signs (+) in front of line items. Indent one or more items to create a nested list. If you need to start an unordered list item with a number followed by a period, you can use a backslash (\) to escape the period.
10. Adding Elements in Lists: To add another element in a list while preserving the continuity of the list, indent the element four spaces or one tab, as shown in the following examples.
11. Fenced Code Blocks: The basic Markdown syntax allows you to create code blocks by indenting lines by four spaces or one tab. If you find that inconvenient, try using fenced code blocks. Depending on your Markdown processor or editor, you’ll use three backticks or three tildes on the lines before and after the code block. The best part? You don’t have to indent any lines.
12. Horizontal Rules: To create a horizontal rule, use three or more asterisks (***), dashes (---), or underscores (___) on a line by themselves.
13. Links: To create a link, enclose the link text in brackets (e.g., [Duck Duck Go]) and then follow it immediately with the URL in parentheses (e.g., (https://duckduckgo.com)).
14. URLs and Email Addresses: To quickly turn a URL or email address into a link, enclose it in angle brackets.
15. Formatting Links: To emphasize links, add asterisks before and after the brackets and parentheses. To denote links as code, add backticks in the brackets.
16. Images: To add an image, add an exclamation mark (!), followed by alt text in brackets, and the path or URL to the image asset in parentheses. You can optionally add a title in quotation marks after the path or URL.
17. Linking Images: To add a link to an image, enclose the Markdown for the image in brackets, and then add the link in parentheses.
18. Escaping Characters: To display a literal character that would otherwise be used to format text in a Markdown document, add a backslash in front of the character.
19. HTML: Many Markdown applications allow you to use HTML tags in Markdown-formatted text. This is helpful if you prefer certain HTML tags to Markdown syntax. For example, some people find it easier to use HTML tags for images. Using HTML is also helpful when you need to change the attributes of an element, like specifying the color of text or changing the width of an image. For security reasons, not all Markdown applications support HTML in Markdown documents. When in doubt, check your Markdown application’s documentation. Some applications support only a subset of HTML tags.
20. Linking to Heading IDs: You can link to headings with custom IDs in the file by creating a standard link with a number sign (#) followed by the custom heading ID. These are commonly referred to as anchor links.
21. Check below examples for more details:


## Blockquotes


> To create a blockquote, add a > in front of a paragraph.  
The BLOCKQUOTE tag is used to display the long quotations (a section that is quoted from another source). It changes the alignment to make it unique from others. In blockquote tag, we can use elements like heading, list, paragraph, etc.  
Block quotations are used for long quotations. The Chicago Manual of Style recommends using a block quotation when extracted text is 100 words or more, or approximately six to eight lines in a typical manuscript.


> Blockquotes can contain multiple paragraphs. Add a > on the blank lines between the paragraphs.
>
> Para-1
>
> Para-2


> Blockquotes can be nested. Add a >> in front of the paragraph you want to nest.
>
>> Para-1


> Blockquotes can contain other Markdown formatted elements. Not all elements can be used — you’ll need to experiment to see which ones work. Example
> #### The quarterly results look great!
>
> - Revenue was off the chart.
> - Profits were higher than ever.
>
>  *Everything* is going according to **plan**.


## Custom Heading/Content
- The World Wide Web Consortium (W3C) has listed 16 valid color names for HTML and CSS:
- Colors: aqua, black, blue, fuchsia, gray, green, lime, maroon, navy, olive, purple, red, silver, teal, white, and yellow.


<span style="font-family:Arial; font-weight:bold; font-size:1.5em; color:aqua;">Custom - Heading/Content using HTML</span>


## Internal Links
To link to a section within your notebook, use the following code formats:
For the text inside the parentheses, replace any spaces and special characters with a hyphen. Applicable for headings only.  
Avoid the double hypens.


1. [Heading Section-1](#Heading-Section-1)
2. [Heading Section-2](#Heading-Section-2)


### Heading Section-1


### Heading Section-2


## Monospace Font


`You can use the monospace font for file paths, file names, message text that users see, or text that users enter.`


## Strikethrough


~~Scratch this~~


## Backslash Escape
Backslash Escape prevents Markdown from interpreting a character as an instruction, rather than as the character itself.


\# Not a header


## Cell Background Color


<code style="background:yellow; color:black">Useful for highlighting to grab the attention of the reader towards certain points.</code>


## HTML Mark Tag
Highlight parts of a text:


Do not forget to buy <mark>milk</mark> today.


## Navigation Menu
It defines a set of navigation links.


<nav>
<a href="https://www.google.com">Google</a> |
<a href="https://www.kaggle.com">Kaggle</a> |
<a href="https://github.com/">GitHub</a>
</nav>


## Underline
Underlined text is not something you typically see in web writing, probably because underlined text is nearly synonymous with links. However, if you’re writing a paper or a report, you may need the ability to underline words and phrases.


Some of these words <ins>will be underlined</ins>.


## Center
Having the ability to center text is a necessity when writing a paper or a report


<center>This text is centered.</center>


## Color
Markdown doesn’t allow you to change the color of text, but if your Markdown processor supports HTML, you can use the <font> HTML tag. The color attribute allows you to specify the font color using a color’s name or the hexadecimal #RRGGBB code.


<font color="red">This text is red!</font>


## Images
To add an image, add an exclamation mark (!), followed by alt text in brackets, and the path or URL to the image asset in parentheses. You can optionally add a title in quotation marks after the path or URL.


Using the image in the same folder as the notebook

```python
# Get the current working directory
import os
cwd=os.getcwd()
cwd
```

Using Markdown Cell


![Alt Text](mlops.webp "Image Title")


Using python code (embedding an image in a code cell): A code cell can also be used to embed images. To display the image, the Ipython.display() method necessitates the use of a function. In the notebook, you can also specify the width and height of the image.

```python
# Import image module
from IPython.display import Image
  
# Get the image
Image(url="mlops.webp", width=400, height=300)
```

Generative adversarial networks (GANs) are like the mischievous pranksters of neural networks, whipping up data out of thin air. They’re so good at it, they can create photos of people’s faces that are so realistic, you’ll swear they’re real – but surprise! Those folks are as imaginary as your diet on cheat day.

```python
# This website shows faces generated by a GAN architecture called StyleGAN:
# Everytime you run the cell; It would be a new face which do not exsit in real life
Image(url="https://thispersondoesnotexist.com/", width=400, height=300)
```

<!-- #region -->
## Videos
We are using the src from the embed link like:
```html
<iframe width="560" height="315" src="https://www.youtube.com/embed/ByIZIKFmHOA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
```
<!-- #endregion -->

```python
# Import the YouTubeVideo
from IPython.display import YouTubeVideo

# Get the Video ID
# https://youtu.be/RBSUwFGa6Fk?si=m_-52Ez3p7jPUnup
YouTubeVideo("RBSUwFGa6Fk", width=560, height=315)
```

```python
# Align the Video
from IPython.display import HTML

HTML("""
<div align="center">
    <iframe width="560" height="315"
    src="https://www.youtube.com/embed/RBSUwFGa6Fk?si=-xJim4lBF03Fyq7a"
    </iframe>
</div>
""")
```

## Colored Boxes


Blue boxes (alert-info)


<div class="alert alert-block alert-info">
<b>Tip:</b> Use blue boxes (alert-info) for tips and notes. 
If it’s a note, you don’t have to include the word “Note”.
</div>


Yellow boxes (alert-warning)


<div class="alert alert-block alert-warning">
<b>Example:</b> Use yellow boxes for examples that are not 
inside code cells, or use for mathematical formulas if needed.
</div>


Green boxes (alert-success)


<div class="alert alert-block alert-success">
<b>Up to you:</b> Use green boxes sparingly, and only for some specific 
purpose that the other boxes can't cover. For example, if you have a lot 
of related content to link to, maybe you decide to use green boxes for 
related links from each section of a notebook.
</div>


Red boxes (alert-danger)


<div class="alert alert-block alert-danger">
<b>Just don't:</b> In general, avoid the red boxes. These should only be
used for actions that might cause data loss or another major issue.
</div>


LaTeX: To be completed later
