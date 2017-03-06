"""Helper function to help generate code-like HTML file"""

def genHTML_header():
	HTML = '<html>\n<head>\n<style type=\"text/css\">\n.sikuli-code {font-size: 20px;font-family: "Osaka-mono", Monospace;line-height: 1.5em; \
            display:table-cell;white-space: pre-wrap;white-space: -moz-pre-wrap !important;white-space: -pre-wrap;white-space: -o-pre-wrap; \
            word-wrap: break-word;width: 99%;} \n \
         .sikuli-code img {vertical-align: middle;margin: 2px;border: 1px solid #ccc;padding: 2px;-moz-border-radius: 5px;-webkit-border-radius: 5px; \
            -moz-box-shadow: 1px 1px 1px gray;-webkit-box-shadow: 1px 1px 2px gray;} \n \
         .kw { \
            color: blue; \
         } \n\
         .skw { \
            color: rgb(63, 127, 127); \
         } \n\
         .str { \
            color: rgb(128, 0, 0); \
         }\n \
         .dig { \
            color: rgb(128, 64, 0); \
         }\n \
         .cmt { \
            color: rgb(200, 0, 200); \
         }\n \
         h2 { \
            display: inline; \
            font-weight: normal; \
         }\n \
         .info { \
            border-bottom: 1px solid #ddd; \
            padding-bottom: 5px; \
            margin-bottom: 20px; \
            display: none; \
         }\n \
         a { \
            color: #9D2900; \
         }\n \
         body { \
            font-family: "Trebuchet MS", Arial, Sans-Serif; \
         }\n \
      </style>\n \
   </head>\n \
<body>\n \
<pre class=\"sikuli-code\">\n'



	return HTML
def genHTML_tail(HTML):
	HTML += '</pre>\n</body>\n</html>'

	return HTML

def genHTML_testBODY(HTML):
	HTML += '<span class=\"skw\">find</span>(<img src=\"1456931791485.png\" />)\n<span class=\"skw\">doubleClick</span>(<img src=\"1456931803633.png\" />)'

	return HTML