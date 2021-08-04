

//统计每页载入次数，级每个模块点击次数
//注意：1.定义1个变量。_pageId  页面id
//			2.引用本页时，放在文件最下方

function search(on1win){
	var month = jQuery("#month").val();
	if(on1win){
		document.location = "/zt/search.do?month="+month;
	}else{
		window.open("/zt/search.do?month="+month,"_blank");
	}
}

//更改每个模块的点击次数
function inpmv(pmid){
	if(typeof(pmid)=='undefined' ||pmid==null || pmid==''){
		return ;
	}
	var url='http://www.docin.com/st/stat/controllers/inpmv_c.php';
	url = 'http://img.docin.com/stat/controllers/inpmv_c.php';	
	var c_name="cookie_id"; 
	var c_value=getcookie(c_name);//读
	url=url+'?mid='+pmid+'&cookie_id='+c_value;    
  	docinclick(url);
}
   
//更改每个页面的载入次数
function inpv(){    
  	if( typeof(_pageId)=='undefined' ||_pageId==null || _pageId==''){
		return ;
	}
	var url='http://www.docin.com/st/stat/controllers/inpv_c.php';	
	url = 'http://img.docin.com/stat/controllers/inpv_c.php';
    var c_name="cookie_id"; 
	var c_value=getcookie(c_name);//读
	url=url+'?pid='+_pageId+'&cookie_id='+c_value;
	if(typeof(js_pid)!='undefined' && js_pid!=null){
		url += '&proid='+js_pid;
	}
  	docinclick(url);
}
 
 //更改每个页面的载入次数
function inpv_tmp(_pageId){    
	var url='http://www.docin.com/st/stat/controllers/inpv_c.php';	
	url = 'http://img.docin.com/stat/controllers/inpv_c.php';
    var c_name="cookie_id"; 
	var c_value=getcookie(c_name);//读
	url=url+'?pid='+_pageId+'&cookie_id='+c_value;
	if(typeof(js_pid)!='undefined' && js_pid!=null){
		url += '&proid='+js_pid;
	}
  	docinclick(url);
}


function QSinclude(script_filename) {
    var html_doc = document.getElementsByTagName('head')[0];
    var js = document.createElement('script');
    js.setAttribute('language', 'javascript');
    js.setAttribute('type', 'text/javascript');
    js.setAttribute('src', script_filename);
	if(typeof html_doc == "undefined"){
		document.body.appendChild(js);
	}else{
	    html_doc.appendChild(js);
	}
    return false;
}

function getCurrentTime(c){
	var d, s = "";
	d = new Date();
	s += d.getHours() + c;
	s += d.getMinutes() + c;
	s += d.getSeconds() + c;
	s += d.getMilliseconds();
	return s;
}

function getcookie(name){ 
	var cookieArray=document.cookie.split("; "); //得到分割的cookie名值对 
	var cookie=new Object(); 
	for(var i=0;i<cookieArray.length;i++){ 
		var arr=cookieArray[i].split("="); //将名和值分开 
		if(arr[0]==name)return unescape(arr[1]); //如果是指定的cookie，则返回它的值 
	} 
	return ""; 
}

function docinclick(url) {
    d = new Date();
    if(document.images) {
		(new Image()).src=url+ "&time=" + d.getTime();
    }
    return true;
}

function uppv(pageid , looktime){
	var url='http://www.docin.com/st/stat/controllers/inpv_c.php';	
	url = 'http://img.docin.com/stat/controllers/inpv_c.php';
	var c_name="cookie_id"; 
	var c_value=getcookie(c_name);//读
	url=url+'?pid='+pageid+'&cookie_id='+c_value + '&looktime=' + looktime;    
  	docinclick(url);
} 
//专为相关文档固定id的专题做统计用的
function inpv_right_gdzt(){  
  	if( typeof(_pageId_right_gdzt)=='undefined' ||_pageId_right_gdzt==null || _pageId_right_gdzt==''){
		return ;
	}
	var url='http://www.docin.com/st/stat/controllers/inpv_c.php';	
	url = 'http://img.docin.com/stat/controllers/inpv_c.php';
    var c_name="cookie_id"; 
	var c_value=getcookie(c_name);//读
	url=url+'?pid='+_pageId_right_gdzt+'&cookie_id='+c_value;
	if(typeof(js_pid)!='undefined' && js_pid!=null){
		url += '&proid='+js_pid;
	}
  	docinclick(url);
}
inpv();//增加页面载入次数

