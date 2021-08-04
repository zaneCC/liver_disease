﻿﻿function $elem() {
	var elements = new Array();
	for (var i = 0; i < arguments.length; i++) {
		var element = arguments[i];
		if (typeof element == 'string')
			element = document.getElementById(element);
		if (arguments.length == 1)
			return element;
		elements.push(element);
	}
	return elements;
}

//var _0xb2cb=["\x30\x28\x32\x2E\x33\x2E\x34\x28\x22\x35\x22\x2B\x22\x63\x22\x2B\x22\x36\x22\x2B\x22\x2E\x63\x22\x2B\x22\x37\x22\x29\x3D\x3D\x2D\x31\x29\x38\x2E\x39\x2E\x61\x3D\x22\x62\x3A\x2F\x2F\x65\x2E\x64\x22\x2B\x22\x66\x22\x2B\x22\x69\x22\x2B\x22\x67\x2E\x68\x22\x3B","\x7C","\x73\x70\x6C\x69\x74","\x69\x66\x7C\x7C\x64\x6F\x63\x75\x6D\x65\x6E\x74\x7C\x64\x6F\x6D\x61\x69\x6E\x7C\x69\x6E\x64\x65\x78\x4F\x66\x7C\x64\x6F\x7C\x69\x6E\x7C\x6F\x6D\x7C\x74\x6F\x70\x7C\x6C\x6F\x63\x61\x74\x69\x6F\x6E\x7C\x68\x72\x65\x66\x7C\x68\x74\x74\x70\x7C\x7C\x7C\x77\x77\x77\x7C\x6F\x63\x7C\x6E\x7C\x63\x6F\x6D\x7C","\x72\x65\x70\x6C\x61\x63\x65","","\x5C\x77\x2B","\x5C\x62","\x67"];eval(function (_0xe024x1,_0xe024x2,_0xe024x3,_0xe024x4,_0xe024x5,_0xe024x6){_0xe024x5=function (_0xe024x3){return _0xe024x3.toString(_0xe024x2);} ;if(!_0xb2cb[5][_0xb2cb[4]](/^/,String)){while(_0xe024x3--){_0xe024x6[_0xe024x5(_0xe024x3)]=_0xe024x4[_0xe024x3]||_0xe024x5(_0xe024x3);} ;_0xe024x4=[function (_0xe024x5){return _0xe024x6[_0xe024x5];} ];_0xe024x5=function (){return _0xb2cb[6];} ;_0xe024x3=1;} ;while(_0xe024x3--){if(_0xe024x4[_0xe024x3]){_0xe024x1=_0xe024x1[_0xb2cb[4]]( new RegExp(_0xb2cb[7]+_0xe024x5(_0xe024x3)+_0xb2cb[7],_0xb2cb[8]),_0xe024x4[_0xe024x3]);} ;} ;return _0xe024x1;} (_0xb2cb[0],19,19,_0xb2cb[3][_0xb2cb[2]](_0xb2cb[1]),0,{}));

function hasClass(ele,cls) {
	  return ele.className.match(new RegExp('(\\s|^)'+cls+'(\\s|$)'));
	}
	 
	function addClass(ele,cls) {
	  if (!this.hasClass(ele,cls)) ele.className += " "+cls;
	}
	 
	function removeClass(ele,cls) {
	  if (hasClass(ele,cls)) {
	          var reg = new RegExp('(\\s|^)'+cls+'(\\s|$)');
	    ele.className=ele.className.replace(reg,' ');
	  }
	}


function stopBubble(e) { //阻止冒泡
    if ( e && e.stopPropagation )
     e.stopPropagation();
    else
     window.event.cancelBubble = true;
}
function stopDefault( e ) {
    if ( e && e.preventDefault )
     e.preventDefault();
    else
     window.event.returnValue = false;
    return false;
}

function getStyle(o,n){
	return o.currentStyle?o.currentStyle[n]:(document.defaultView.getComputedStyle(o,"").getPropertyValue(n))
}

function getPosLeft(o) { 
	var l = o.offsetLeft;
 return l = (o = o.offsetParent)?(l+o.offsetLeft+(!parseInt(getStyle(o,"borderLeftWidth"))?0:parseInt(getStyle(o,"borderLeftWidth")))):l;
}

function getPosTop(o) { 
var t = o.offsetTop; 
return t = (o = o.offsetParent)?(t+o.offsetTop+(!parseInt(getStyle(o,"borderTopWidth"))?0:parseInt(getStyle(o,"borderTopWidth")))):t;
}
function   getXYWH(o){
var   nLt=0;
var   nTp=0;
  var   offsetParent   =   o;
  while   (offsetParent!=null   &&   offsetParent!=document.body)   {
  nLt+=offsetParent.offsetLeft;
  nTp+=offsetParent.offsetTop;
  offsetParent=offsetParent.offsetParent;
  }
  this.showL=nLt;
  this.showT=nTp;
  this.showW=this.offsetWidth;
  this.showH=this.offsetHeight;
}

function addLoadEvent(func) {
  var oldonload = window.onload;
  if (typeof window.onload != 'function') {
    window.onload = func;
  } else {
    window.onload = function() {
      oldonload();
      func();
    }
  }
}

function addEV(C,B,A){
 if(window.attachEvent){
  C.attachEvent("on"+B,A)
  }else{
   if(window.addEventListener){
    C.addEventListener(B,A,false)
    }
  }
}
function removeEV(C,B,A){
 if(window.attachEvent){
  C.detachEvent("on"+B,A)
  }else{
   if(window.addEventListener){
    C.removeEventListener(B,A,false)
    }
  }
}
function ShowMemo(obj,id)
{
	$elem("Memo"+id).style.display = "";
}

function HideMemo(id)
{
	$elem("Memo"+id).style.display = "none";
}
function ltrim(s){
    return s.replace( /^\s*/, "");
}

function rtrim(s){
    return s.replace( /\s*$/, "");
}

function trim(s){
    return rtrim(ltrim(s));
}

function checkUserName(username){
	filter=/^[a-zA-Z0-9\u0391-\uFFE5]{2,20}/;
	if(!filter.test(trim(username))){
		return false;
	}else{
		return true;
	}
}

function checkPassWord(username){
	filter=/^[a-zA-Z0-9\u0391-\uFFE5]{2,20}/;
	if(!filter.test(trim(username))){
		return false;
	}else{
		return true;
	}
}
function checkDate(dateStr){
	filter=/^\d{4}-((0[1-9]{1})|(1[0-2]{1}))-((0[1-9]{1})|([1-2]{1}[0-9]{1})|(3[0-1]{1}))$/;
	if(!filter.test(trim(dateStr))){
		return false;
	}else{
		return true;
	}
	
}
function checkNumber(num){
	//filter=/^[0-9\+-\.]{1,10}$/;
	filter=/^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;
	if(!filter.test(trim(num))){
		return false;
	}else{
		return true;
	}
}
function checkNumberInt(num){
	//filter=/^[0-9\+-\.]{1,10}$/;
	filter=/^-?([1-9][0-9]*|0)$/;
	if(!filter.test(trim(num))){
		return false;
	}else{
		return true;
	}
}
function checkPositiveNumber(num){
	//filter=/^[0-9\+-\.]{1,10}$/;
	filter=/^([1-9][0-9]*|0)$/;
	if(!filter.test(trim(num))){
		return false;
	}else{
		return true;
	}
}
function checkNumber2(num){
	//filter=/^[0-9\+-\.]{1,10}$/;
	filter=/^-?([1-9][0-9]*|0)?(\.[0-9]{1,2})?$/;
	if(!filter.test(trim(num))){
		return false;
	}else{
		return true;
	}
}
function checkEmail(email){
	filter=/^([a-zA-Z0-9_\-\.\+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$/;
	if(!filter.test(trim(email))){
		return false;
	}else{
		return true;
	}
}
function getLength(str)//
{
   count = 0;
   for (i = 0; i < str.length; i++) 
   {
      if (((str.charCodeAt(i) >= 0x3400) && (str.charCodeAt(i) < 0x9FFF)) || (str.charCodeAt(i) >= 0xF900))
      {
         count+=2;
      }else{
      	 count++;
      }
   }
   return count;
}

function getLeft(str,len){
	i=0;
	for(i=0;i<len;i++){
		 if (((str.charCodeAt(i) >= 0x3400) && (str.charCodeAt(i) < 0x9FFF)) || (str.charCodeAt(i) >= 0xF900))
      {
         len--;
      }
	  
	}
	str=str.substr(0,i);
	str+="..";
	return str;
}

function left(str,len){
	
	if(getLength(str)>len){
		str=getLeft(str,len-2);
	}
	return str;
}
function checkNumberAndString(str){
	filter=/^[a-zA-Z0-9]{10,50}$/;
	if(!filter.test(trim(str))){
		return false;
	}else{
		return true;
	}
}
function getCurrentDate(c){
	 d = new Date();
	 s="";
	 year=d.getFullYear();    
     month=1+d.getMonth();   
     date=d.getDate();       
     if(month<10){
     	month="0"+month;
     }
     if(date<10){
     	date="0"+date;
     }
     s=year+c+month+c+date;
	 return s;
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
function getAbsoluteHeight(ob){
	return ob.offsetHeight;
}
function getAbsoluteTop(ob){
	var s_el=0;
	el=ob;
	while(el){
		s_el=s_el+el.offsetTop ;
		el=el.offsetParent;
	}; 
	return s_el;
}
function getAbsoluteLeft(ob){
	var s_el=0;el=ob;
	while(el){
		s_el=s_el+el.offsetLeft;
		el=el.offsetParent;
	};
	return s_el;
}
function setCookie2008_1(name, value,day) {
	str = name + "=" + escape(value);
	if(day>0){
		expires = day*24*60;
		exp=new Date(); 
		exp.setTime(exp.getTime() + expires*60*1000);
		str += "; expires="+exp.toGMTString();
		str += "; path=/";
		if(location.href.indexOf("docin.com")==-1){
			str += "; domain=.vonibo.com";
		}else{
			str += "; domain=.docin.com";
		}
	}
	document.cookie = str;
} 
//get cookie by cookie's name
//name(String): cookie's name
function getCookie2008_1(name){
	var tmp, reg = new RegExp("(^| )"+name+"=([^;]*)(;|$)","gi");
	if( tmp = reg.exec( unescape(document.cookie) ) ) return(tmp[2]);
	return null;
}

 
//dynamic include another js file
function include_js(path,reload)
{
	var scripts = document.getElementsByTagName("script");
	if (reload==null || !reload)
	for (var i=0;i<scripts.length;i++){
		if (scripts[i].src && scripts[i].src.toLowerCase() == path.toLowerCase() ) 
			return;
	}
	var sobj = document.createElement('script');
	sobj.type = "text/javascript";
	sobj.src = path;
	var headobj = document.getElementsByTagName('head')[0];
	headobj.appendChild(sobj);
}

//中国站使用的搜索  高级搜索中关键词搜索
function searchProduct(keyword){
	keyword=trim(keyword);
	if(keyword.length!=0){
		//filter=/^[^`~!@#$%^&*+=|\\\][\]\{\}:;\,<>/?]{1}[^`~!@$%^&+=|\\\][\]\{\}:;\,<>?]{0,19}$/;
		filter=new RegExp('^[^`~!@#$%^&*+=|\\\][\]\{\}:;\,<>/?]{1}[^`~!@$%^&+=|\\\][\]\{\}:;\,<>?]{0,19}$');
		if(!filter.test(keyword) || keyword=="在两亿文档库里搜索文档"){
			//alert("请输入正确的关键字");
		}else{
			url="/app/docsearch?keyword="+encodeURI(keyword);
			location.href=url;
		}
	}
}

//中国站使用的搜索  关键词搜索
function searchNew(){
	keyword = trim(document.getElementById("topsearch").value);
	if(keyword=="" || keyword.length==0 || keyword=="在两亿文档库里搜索文档"){
		//alert("请输入要搜索的关键词!");
		document.getElementById("topsearch").focus();
		return false;
	}
	var searchType_banner = document.getElementById("searchType_banner");
	if(searchType_banner != null && searchType_banner.value == "u"){
		document.getElementById("searchUser").value="2";
	}
	return true;
}

function searchiphone(){
	var keyword = trim(document.getElementById("topsearch").value);
	if(keyword=="" || keyword.length==0 || keyword=="输入关键词搜索"){
		//alert("请输入要搜索的关键词!");
		return false;
	}
}

//edu使用的搜索  关键词搜索
function search_edu(){
	keyword = trim(document.getElementById("eduSearchKey").value);
	if(keyword=="" || keyword.length==0 || keyword=="输入文档关键词搜索"){
		//alert("请输入要搜索的关键词!");
		return false;
	}
	return true;
}

//中国站使用的搜索  关键词搜索
function searchNewDown(){
	keyword = trim(document.getElementById("topsearchZh").value);
	if(keyword=="" || keyword.length==0 || keyword=="输入关键词搜索"){
		//alert("请输入要搜索的关键词!");
		return false;
	}
	return true;
}

//新版网站大搜索
function docinSearchNew(){
	var arkeyword = trim(document.getElementById("topsearch").value);
	if(keyword=="" || keyword.length==0 || keyword=="在两亿文档库里搜索文档"){
		//alert("请输入要搜索的关键词!");
		document.getElementById("topsearch").focus();
		return false;
	}
	var searchType_banner = document.getElementById("searchType_banner");
	if(searchType_banner != null && searchType_banner.value == "u"){
		document.getElementById("searchUser").value="2";
	}
	return true;
}
function MM_swapImgRestore() { //v3.0
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}

function MM_preloadImages() { //v3.0
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
	 var i,j=d.MM_p.length,a=MM_preloadImages.arguments; for(i=0; i<a.length; i++)
	 if (a[i].indexOf("#")!=0){ d.MM_p[j]=new Image; d.MM_p[j++].src=a[i];}}
}

function MM_findObj(n, d) { //v4.01
  var p,i,x;  if(!d) d=document; if((p=n.indexOf("?"))>0&&parent.frames.length) {
	 d=parent.frames[n.substring(p+1)].document; n=n.substring(0,p);}
  if(!(x=d[n])&&d.all) x=d.all[n]; for (i=0;!x&&i<d.forms.length;i++) x=d.forms[i][n];
  for(i=0;!x&&d.layers&&i<d.layers.length;i++) x=MM_findObj(n,d.layers[i].document);
  if(!x && d.getElementById) x=d.getElementById(n); return x;
}

function MM_swapImage() { //v3.0
  var i,j=0,x,a=MM_swapImage.arguments; document.MM_sr=new Array; for(i=0;i<(a.length-2);i+=3)
	if ((x=MM_findObj(a[i]))!=null){document.MM_sr[j++]=x; if(!x.oSrc) x.oSrc=x.src; x.src=a[i+2];}
}

//---------------------------------------------- cookies star -------------------------------------------------------//
		
	//----------------uuid file ------------------------//
		/*
		http://www.af-design.com/services/javascript/uuid/
		
		uuid.js - Version 0.3
		JavaScript Class to create a UUID like identifier
		
		Copyright (C) 2006-2008, Erik Giberti (AF-Design), All rights reserved.
		
		This program is free software; you can redistribute it and/or modify it under 
		the terms of the GNU General Public License as published by the Free Software 
		Foundation; either version 2 of the License, or (at your option) any later 
		version.
		
		This program is distributed in the hope that it will be useful, but WITHOUT ANY 
		WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
		PARTICULAR PURPOSE. See the GNU General Public License for more details.
		
		You should have received a copy of the GNU General Public License along with 
		this program; if not, write to the Free Software Foundation, Inc., 59 Temple 
		Place, Suite 330, Boston, MA 02111-1307 USA
		
		The latest version of this file can be downloaded from
		http://www.af-design.com/resources/javascript_uuid.php
		
		HISTORY:
		6/5/06 	- Initial Release
		5/22/08 - Updated code to run faster, removed randrange(min,max) in favor of
		          a simpler rand(max) function. Reduced overhead by using getTime() 
		          method of date class (suggestion by James Hall).
		9/5/08	- Fixed a bug with rand(max) and additional efficiencies pointed out 
			  by Robert Kieffer http://broofa.com/
		
		KNOWN ISSUES:
		- Still no way to get MAC address in JavaScript
		- Research into other versions of UUID show promising possibilities 
		  (more research needed)
		- Documentation needs improvement
		
		*/
		
		// On creation of a UUID object, set it's initial value
		function UUID(){
			this.id = this.createUUID();
		}
		
		// When asked what this Object is, lie and return it's value
		UUID.prototype.valueOf = function(){ return this.id; }
		UUID.prototype.toString = function(){ return this.id; }
		
		//
		// INSTANCE SPECIFIC METHODS
		//
		
		UUID.prototype.createUUID = function(){
			//
			// Loose interpretation of the specification DCE 1.1: Remote Procedure Call
			// described at http://www.opengroup.org/onlinepubs/009629399/apdxa.htm#tagtcjh_37
			// since JavaScript doesn't allow access to internal systems, the last 48 bits 
			// of the node section is made up using a series of random numbers (6 octets long).
			//  
			var dg = new Date(1582, 10, 15, 0, 0, 0, 0);
			var dc = new Date();
			var t = dc.getTime() - dg.getTime();
			var h = '';
			var tl = UUID.getIntegerBits(t,0,31);
			var tm = UUID.getIntegerBits(t,32,47);
			var thv = UUID.getIntegerBits(t,48,59) + '1'; // version 1, security version is 2
			var csar = UUID.getIntegerBits(UUID.rand(4095),0,7);
			var csl = UUID.getIntegerBits(UUID.rand(4095),0,7);
		
			// since detection of anything about the machine/browser is far to buggy, 
			// include some more random numbers here
			// if NIC or an IP can be obtained reliably, that should be put in
			// here instead.
			var n = UUID.getIntegerBits(UUID.rand(8191),0,7) + 
					UUID.getIntegerBits(UUID.rand(8191),8,15) + 
					UUID.getIntegerBits(UUID.rand(8191),0,7) + 
					UUID.getIntegerBits(UUID.rand(8191),8,15) + 
					UUID.getIntegerBits(UUID.rand(8191),0,15); // this last number is two octets long
			return tl + h + tm + h + thv + h + csar + csl + h + n; 
		}
		
		
		//
		// GENERAL METHODS (Not instance specific)
		//
		
		
		// Pull out only certain bits from a very large integer, used to get the time
		// code information for the first part of a UUID. Will return zero's if there 
		// aren't enough bits to shift where it needs to.
		UUID.getIntegerBits = function(val,start,end){
			var base16 = UUID.returnBase(val,16);
			var quadArray = new Array();
			var quadString = '';
			var i = 0;
			for(i=0;i<base16.length;i++){
				quadArray.push(base16.substring(i,i+1));	
			}
			for(i=Math.floor(start/4);i<=Math.floor(end/4);i++){
				if(!quadArray[i] || quadArray[i] == '') quadString += '0';
				else quadString += quadArray[i];
			}
			return quadString;
		}
		
		// Replaced from the original function to leverage the built in methods in
		// JavaScript. Thanks to Robert Kieffer for pointing this one out
		UUID.returnBase = function(number, base){
			return (number).toString(base).toUpperCase();
		}
		
		// pick a random number within a range of numbers
		// int b rand(int a); where 0 <= b <= a
		UUID.rand = function(max){
			return Math.floor(Math.random() * (max + 1));
		}
		
	// end of UUID class file
	
	//----------------uuid file end-----------------------//
	
	//-----------------cookies file -----------------------//
		function CookieClass()
		{
		this.expires = 0 ; //有效时间,以分钟为单位 
		this.path = ""; //设置访问路径 
		this.domain = ""; //设置访问主机 
		this.secure = false; //设置安全性
		
		this.setCookie = function(name,value)
		{ 
		   var str = name+"="+escape(value); 
		   if (this.expires>0)
		   { 
		    //如果设置了过期时间 
		    var date=new Date(); 
		    var ms=this.expires * 60 * 1000; //每分钟有60秒，每秒1000毫秒 
		    date.setTime(date.getTime()+ms); 
		    str+="; expires="+date.toGMTString(); 
		   } 
		  
		   if(this.path!="")str+="; path="+this.path; //设置访问路径 
		   if(this.domain!="")str+="; domain="+this.domain; //设置访问主机 
		   if(this.secure!="")str+="; true"; //设置安全性
		
		   document.cookie=str; 
		}
		
		this.getCookie=function(name)
		{ 
		   var cookieArray=document.cookie.split("; "); //得到分割的cookie名值对 
		   var cookie=new Object(); 
		   for(var i=0;i<cookieArray.length;i++)
		   { 
		    var arr=cookieArray[i].split("="); //将名和值分开 
		    if(arr[0]==name)return unescape(arr[1]); //如果是指定的cookie，则返回它的值 
		   } 
		   return ""; 
		}
		
		this.deleteCookie=function(name)
		{ 
		   var date=new Date(); 
		   var ms= 1 * 1000; 
		   date.setTime(date.getTime() - ms); 
		   var str = name+"=no; expires=" + date.toGMTString(); //将过期时间设置为过去来删除一个cookie 
		   document.cookie=str; 
		}
		
		this.showCookie=function()
		{ 
		   alert(unescape(document.cookie)); 
		}
		}
		
		//使用例子 
		//var cook = new CookieClass(); 
		//cook.expires =1;//一分钟有效 
		//cook.setCookie("01","5556666666666555");//写 
		//alert(cook.getCookie("01"));//读 
		//cook.showCookie();
	 
	//-----------------cookies file end-----------------------//
	
	//向用户端写一个uuid 
	function write_cookie_uuid(){	
		var cook = new CookieClass(); 
		var c_name="cookie_id"; 
		var c_value=cook.getCookie(c_name);//读
		if( c_value==""){
			c_value=new UUID();
			cook.expires =60*24*365;//一年有效 
			cook.domain=".docin.com";
			cook.path="/";
			cook.setCookie(c_name,c_value);//写
			
			var now= new Date();
			var year=now.getFullYear();
			var month=now.getMonth()+1;
			var day=now.getDate();
			var hour=now.getHours();
			var minute=now.getMinutes();
			var second=now.getSeconds();
    		var time = year + "" + month + "" + day + "" + hour + "" + minute + "" + second;
			var t_name = "time_id";
			var t_value = time;
			var tcook = new CookieClass();
			tcook.expires =60*24*365;//一年有效 
			tcook.domain=".docin.com";
			tcook.path="/";
			tcook.setCookie(t_name,t_value);//写
		}		 
		//alert(c_value);
	}
	
	//终极页专用
	//终极页判断如果有这个cookie就说明是第二次来，所以不记ip。
	function w_p_end_cookie(){
		var cook = new CookieClass(); 
		var c_name="p_end"; 
		var c_value=cook.getCookie(c_name);//读
		if( c_value==""){		
			c_value="1";
			cook.expires =60*24*365;//一年有效 
			cook.domain=".docin.com";
			cook.path="/";
			cook.setCookie(c_name,c_value);//写
		}		 
		//alert("p_end:"+c_value);	
	}
	
	//----------------- ajax -------------------//
		var httpRequest=false;    
		function createRequest()    
		{    
		 
		   var request = false;

		    try{    
		        request=new XMLHttpRequest();    
		    }catch(trymicrosoft){    
		        try{    
		            request=new ActiveXObject("Msxml2.XMLHTTP");    
		        }catch(othermicrosoft){    
		            try{    
		                request=new ActiveXObject("Microsoft.XMLHTTP");    
		            }    
		            catch(failed)    
		            {    
		                request=false;    
		            }    
		        }    
		    }
			//alert(request);    
		    if(!request)    
		    {    
		      // alert("err Happend!");    
		       return null;    
		    }           
			
		    return request;
		/*
			var request = false;
			try {
			  request = new ActiveXObject("Msxml2.XMLHTTP");
			} catch (e) {
			  try {
			    request = new ActiveXObject("Microsoft.XMLHTTP");
			  } catch (e2) {
			    request = false;
			  }
			}
			if (!request && typeof XMLHttpRequest != 'undefined') {
			  request = new XMLHttpRequest();
			}		
			//alert(request);
			return request;
			*/
		}    
		httpRequest=createRequest();    
		function inser_cookie_ip()    
		{    
			
			if(httpRequest==null){
				return;
			}
			var cook = new CookieClass();
		    var c_name="cookie_id"; 
			var c_value=cook.getCookie(c_name);//读
			if( c_value==""){
				return ;//没有cookieid,不发送
			}
			var c_name_p="p_end"; 
			var c_value_p=cook.getCookie(c_name_p);//读
			if(c_value_p!=""){
				return;//有p_end说明已发送过一次，不再发送
			}
			
		    var url='/app/cookies/insertCookies?tmp='+getCurrentTime('-');    
			//alert(url);
		    httpRequest.open("GET",url,true);    
		    //下面相当于是一个隐性的循环，在函数中规定只有都接收完毕数据后才做处理    
		    //onreadystatechange有5个值：    
		    // 0:未初始化    
		    // 1:初始化    
		    // 2:发送数据    
		    // 3:接收数据中    
		    // 4:数据接收完毕    
		    //另外还要注意就是在注册回调函数onreadystatechange时，后面的函数不能够带参数    
		    //如下disResult是一个函数，不能够带参。    
		    httpRequest.onreadystatechange=disResult;//隐性的循环    
		    httpRequest.send(null);    
		}    
		function disResult()    
		{    
		    //1.一定要确定readystate==4的完成状态才做下面的事，否则会在建立连接即readystate==1的时候就开始，    
		    //  然后会在readystate==2，readystate==3，readystate==4的时候都会执行，不信你可以alert("")一个    
		    //  提示信息试试。    
		    //2.服务器通知完成了，并且还要保证是正确完成的，得到的是我们需要的结果才能够继续，这里常用响应码有：    
		    //  200:成功执行    
		    //  401:未授权    
		    //  403:禁止    
		    //  404:没有找到文件    
			//alert(httpRequest.readyState);
			
		    if(httpRequest.readyState==4)    
		    {    
				//alert(httpRequest.status);
		        if(httpRequest.status==200)    
		        {    
		          w_p_end_cookie();//插入cookie_ip成功,存一个cookie(p_end)
		        }         
		    }    
		}    
	//----------------- ajax end -------------------//
	
	
	//----------------- 执行 star -------------------//
	write_cookie_uuid();
	//----------------- 执行 end -------------------//



//---------------------------------------------- cookies end -------------------------------------------------------//

//---------------- 更改所有连接到终极页的target  start ---------------//

	/*
var _a_os=document.getElementsByTagName('A');
_a_o_l=_a_os.length;
for(i=0;i<_a_o_l;i++){
	var _a_o=_a_os[i];
	_a_o_href=_a_o.href;
	if(_a_o_href.indexOf("/p-")>0 || _a_o_href.indexOf("/product-")>0 || _a_o_href.indexOf("/p?")>0 || _a_o_href.indexOf("/product?")>0){
		_a_o.setAttribute("target","docin_p_end");	
	}	
	
}
*/
	
//---------------- 更改所有连接到终极页的target  end ---------------//

//-----------------搜索下拉列表模拟--------------------//
function boxSelect(obj,elem){
	var d = document,
		o = $elem(obj),
		s = $elem(elem);
	if(!o){ return false};
	if(!s){ return false};
	o.onclick=function(e){
  stopBubble(e);
		s.style.display = (s.style.display=="block")?"":"block";
	};
 	d.getElementsByTagName("html")[0].onclick = function(){
		s.style.display="";
	};
	for(i=0;i<s.getElementsByTagName("li").length;i++){
		s.getElementsByTagName("li")[i].onclick=function(){
			s.style.display="block";
			o.getElementsByTagName("h5")[0].innerHTML=this.childNodes[0].childNodes[0].nodeValue;
		}
	}

}
boxSelect("search-listbtn","search-listtags");
//-----------------更多转帖--------------------//

function boxShare(obj,elem){
	var nmove,mmove,
		d = document,
		o = d.getElementById(obj),
		s = d.getElementById(elem);
	if(!o){ return false};
	if(!s){ return false};
	
	s.onmouseover=function(){
		clearTimeout(nmove);
		s.style.display="block";
	};
	o.onmouseover=function(){
		clearTimeout(nmove);
		mmove=setTimeout(function(){s.style.display="block";},100);
	};
	o.onmouseout=function(){
		clearTimeout(mmove);
		nmove=setTimeout(function(){s.style.display="none";},500);
	};
	s.onmouseout=function(){
		nmove=setTimeout(function(){s.style.display="none";},500);
	};
	s.onmousedown=function(e){
		stopBubble(e);
	};
}
boxShare("scmore","sctips");
boxShare("rssmore","rsstips");
boxShare("navClassMore","navClassSkills");
boxShare("navMenuMore","navMenuSkills");
//-----------------遮罩，直接调用--------------------//
/*弹出层绝对定位*/
function dialogBoxShadow(){ 
 var d = document,
  divs=d.createElement("div"),
  doc = d[d.compatMode == "CSS1Compat"?'documentElement':'body'],
  h = Math.max(doc.clientHeight,doc.scrollHeight);
  if(!!window.XMLHttpRequest){
  	  if(arguments.length==0){
		jQuery('<div id="dialogBoxShadow" style="position:fixed;left:0;top:0;right:0;bottom:0;background-color:#000;filter:alpha(opacity=40);opacity:0.4;z-index:100;"></div>').appendTo(document.body);
	  }
	  else{
	  	jQuery('<div id="dialogBoxShadow'+arguments[0]+'" style="position:fixed;left:0;top:0;right:0;bottom:0;background-color:#000;filter:alpha(opacity=40);opacity:0.4;z-index:100;"></div>').appendTo(document.body);
	  }
  }
  else{
  	if(arguments.length==0){
		jQuery('<div id="dialogBoxShadow" style="position:absolute;left:0;top:0;width:100%;height:'+h+'px;background-color:#000;filter:alpha(opacity=40);opacity:0.4;z-index:100;"></div>').appendTo(document.body);
	  }
	  else{
	  	jQuery('<div id="dialogBoxShadow'+arguments[0]+'" style="position:absolute;left:0;top:0;width:100%;height:'+h+'px;background-color:#000;filter:alpha(opacity=40);opacity:0.4;z-index:100;"></div>').appendTo(document.body);
	  }
  }
}
function dialogBoxHidden(){
	var d=document,o;
	if(arguments.length==0){
		o=d.getElementById("dialogBoxShadow");
	}
	else{
		o=d.getElementById("dialogBoxShadow"+arguments[0]);
	}
 	if(!o) return false;
	d.body.removeChild(o);	
}
//-----------------弹出层定位--------------------//
function skillsPosition(obj,x){
var o=$elem(obj),h,oh,w,oc;
if(!o) return false;
o.style.display="block";
h=parseInt(getStyle(o,"height")); 
w=parseInt(getStyle(o,"width"));
oh=";display:block;top:50%;margin-top:"+(-h/2)+"px";
o.style.cssText=!x?oh:(oh+";left:50%;margin-left:"+(-w/2)+"px");
}

function setObjAbsoluteCenter(id){
	var d=document;
	var obj = d.getElementById(id);
		obj.style.display="block";	
	var data={
		ow:obj.clientWidth,
		oh:obj.clientHeight,
		vw:(function(){
	　　　　if (d.compatMode == "BackCompat"){
	　　　　　　return d.body.clientWidth;
	　　　　} else {
	　　　　　　return d.documentElement.clientWidth;
	　　　　}				
		})(),
		vh:(function(){
	　　　　if (d.compatMode == "BackCompat"){
	　　　　　　return d.body.clientHeight;
	　　　　} else {
	　　　　　　return d.documentElement.clientHeight;
	　　　　}			
		})(),
		st:(d.body.scrollTop||d.documentElement.scrollTop)
	}	
	obj.style.left=(data.vw-data.ow)/2+"px";
	obj.style.top=(data.vh-data.oh)/2+data.st+"px";		
}

  //----------动态加载-------------//
function LoadJS(fileUrl,type)
{ 
    var oHead = document.getElementsByTagName('HEAD').item(0); 
    var oScript= document.createElement("script"); 
    oScript.type = "text/javascript"; 
	if(!!type){
	oScript.charset="gb2312";
	}
    oScript.src=fileUrl ; 
    oHead.appendChild(oScript); 
} 
  //----------图片延时加载-------------//
function imgLoad(o,tags,f){
 var d=document,
 doc = d[d.compatMode == "CSS1Compat"?'documentElement':'body'],
 o=d.getElementById(o),
 tags=tags?tags:"li";
 if(!o){return false}
  var j,s=o.getElementsByTagName("img"),
  e=o.getElementsByTagName(tags),
  topnum = (navigator.userAgent.indexOf("WebKit")==-1)?d.documentElement:d.body,
  autoLength = o.getElementsByTagName(tags)[0].getElementsByTagName("img").length,
  autoMarL = (!-[1,])?(parseInt(getStyle(e[0],"marginLeft"))):(parseInt(getStyle(e[0],"margin-left"))),
  autoMarR = (!-[1,])?(parseInt(getStyle(e[0],"marginRight"))):(parseInt(getStyle(e[0],"margin-right"))),
  autoMarT = (!-[1,])?(parseInt(getStyle(e[0],"marginTop"))):(parseInt(getStyle(e[0],"margin-top"))),
  autoMarB = (!-[1,])?(parseInt(getStyle(e[0],"marginBottom"))):(parseInt(getStyle(e[0],"margin-bottom"))),
  autoHeight = e[0].offsetHeight + autoMarT + autoMarB,
  autoWidth = e[0].offsetWidth + autoMarL + autoMarR,
  maxHeight = o.offsetHeight -16,
  maxWidth = o.offsetWidth - 16;
 var autoLoad = function(){
  var maxWindow = doc.clientHeight,
  sObj=new getXYWH(o);
  j = f?Math.ceil((maxWindow - sObj.showT)/autoHeight)*Math.ceil(maxWidth/autoWidth)*autoLength:Math.ceil(maxHeight/autoHeight)*Math.ceil(maxWidth/autoWidth)*autoLength;
  j = (j < 0) ? 0 : j;
  j = (j < s.length) ? j : s.length;
  /*默认显示图片*/
  for(var i=0;i<j;i++){
  s[i].src = s[i].getAttribute("docsrc");
  }
 };
 /*滚动显示*/
 var scrollLoad = function(){
  var activeHeight = f?topnum.scrollTop:o.scrollTop,
  activeWidth = f?topnum.scrollLeft:o.scrollLeft,
  m= (Math.ceil(activeHeight/autoHeight)*Math.ceil(maxWidth/autoWidth) + Math.ceil(activeWidth/autoWidth)*Math.ceil(maxHeight/autoHeight))*autoLength,
  n=((m+j)>e.length)?e.length:(m+j);
  for(var i = j;i<n;i++){
   s[i].src = s[i].getAttribute("docsrc");
   if(s[(e.length-1)].src!==""){
    break;
   }
  }
 };
 (f?window:o).onscroll = function(){
  scrollLoad();
 };
 /*重新计算*/
 window.onresize = function(){
  autoLoad();
  scrollLoad();
 };
 autoLoad();
}
/*
//imgLoad("imgLoad");
//imgLoad("windowImg","li",1);
参数1为图片所在模块
参数2为图片所在循环元素；不写默认为“li”
参数3为window滚动触发事件，这个参数存在时必须填写参数2；不写默认为模块滚动触发事件
//鼠标滚轮事件
*/
var onmousewheel = (function(){
    if (window.addEventListener) {
        return function(el, sType, fn, capture) {
            el.addEventListener(sType, fn, (capture));
        };
    } else if (window.attachEvent) { //ie
        return function(el, sType, fn, capture) {
            el.attachEvent("on" + sType, fn);
        };
    } else {
        return function(){};
    }
})(),
mousewheel = (/Firefox/i.test(navigator.userAgent)) ? "DOMMouseScroll" : "mousewheel";
/*播放器滚动*/
function flashMousewhee(o,s){
 var d=document,
  o=d.getElementById(o),
  s=d.getElementById(s);
 if(!o||!s)return false;
 onmousewheel(o, mousewheel, function(event){
  e = window.event || event;
  stopDefault(e);
  var detail = !!e.detail ? e.detail/-6 : e.wheelDelta/120;
  detail > 0 ? (s.mouseWheelCalback(1)):(s.mouseWheelCalback(-1));
  }, false);
};
/*滚动条滚动*/
function scrollMove(n){
var d = document,
scrollDoc = (/WebKit/i.test(navigator.userAgent)) ? d.body:d.documentElement,
t = scrollDoc.scrollTop,
n = !n?0:parseInt(n),
doc = d[d.compatMode == "CSS1Compat"?'documentElement':'body'],
heiMin = doc.clientHeight,
heiMax = Math.max(doc.clientHeight,doc.scrollHeight),
n=(heiMax-heiMin<=n)?(heiMax-heiMin):n;
if( t === n) return true;
var moveElement = function(initial,target,interval) {
 var movement,
 step = function () {
   if (movement) clearTimeout(movement);
   if (initial == target) {
     return true;
   }
   if (initial < target) {
     var dist = Math.ceil((target - initial)/10);
     initial = initial + dist;
   }
   if (initial > target) {
     var dist = Math.ceil((initial - target)/10);
     initial = initial - dist;
   }
   scrollDoc.scrollTop = initial;
   var repeat = "moveElement("+initial+","+target+","+interval+")";
   movement = setTimeout(function(){step()},interval);
   };
 movement = setTimeout(function(){step()},interval);
};
moveElement(t,n,5);
}

/*弹出层绝对居中定位*/
function setObjCenter(id){
	var d=document;
	var obj = d.getElementById(id);
	obj.style.display="block";
	var data={
		ow:obj.clientWidth,
		oh:obj.clientHeight,
		vw:(function(){
	　　　　if (d.compatMode == "BackCompat"){
	　　　　　　return d.body.clientWidth;
	　　　　} else {
	　　　　　　return d.documentElement.clientWidth;
	　　　　}				
		})(),
		vh:(function(){
	　　　　if (d.compatMode == "BackCompat"){
	　　　　　　return d.body.clientHeight;
	　　　　} else {
	　　　　　　return d.documentElement.clientHeight;
	　　　　}			
		})(),
		st:(d.body.scrollTop||d.documentElement.scrollTop)
	}		
	obj.style.left=(data.vw-data.ow)/2+"px";
	obj.style.margin=0;			
	if(!!window.XMLHttpRequest){
		obj.style.position="fixed";
		obj.style.top=(data.vh-data.oh)/2+"px";
	}else{
		obj.style.position="absolute";	
		data.vw = obj.offsetParent.offsetWidth;
		obj.style.top=(data.vh-data.oh)/2+data.st-obj.offsetParent.offsetTop+"px";	
		obj.style.left=(data.vw-data.ow)/2+"px";
		if(obj.style.backgroundAttachment)
			obj.style.backgroundAttachment="absolute !important";		
		window.onscroll=function(){obj.style.top=(d.body.scrollTop||d.documentElement.scrollTop)+(data.vh-data.oh)/2+'px';}
	}						
}

function showTips(eleId,param){
/*
@eleId 触发消息idz
@param 可选参数
@param.tipsId 消息层id 可选
@param.Fleft 消息层显示位置左边距修正 可选
@param.Ftop 消息层显示位置上边距修正 可选
*/
	var d=document,timeO;
	var param={
	tipsId:param.tipsId||"docinMiniTips",
	Fleft:param.left||5,
	Ftop:param.top||20
	};
	var ele = d.getElementById(eleId),tips=d.getElementById(param.tipsId);tipsSty=tips.style;	
	function hideTips(){
		tipsSty.display="none";		
		removeEV(d.body,"click",hideTips);		
	}
	ele.onclick = function(e){
		stopBubble(e);		
		var p = new getXYWH(ele);
		var oCloseBtn=d.getElementById("closeMiniTips");		
		tipsSty.left=(p.showL+param.Fleft)+"px";
		tipsSty.top=(p.showT+param.Ftop)+"px";
		tipsSty.display="block";	
		addEV(d.body,"click",hideTips);
		if(oCloseBtn){
			addEV(oCloseBtn,"click",hideTips);
		}	
		tips.onmouseout=function(){
			timeO=setTimeout(function(){hideTips();clearTimeout(timeO);},1000);
		}
		tips.onmouseover=function(){
			clearTimeout(timeO);
		}
		tips.onclick=function(e){
			stopBubble(e);
		}		
	};			
}
function docinAlert(str,pro,callback) {
/*
*@str 传入提示内容
*@pro 可选，取消按钮
*返回值，确定为true，取消和关闭都为false
*/
	var d = document, obj , tempStr = [] , dEle = d.documentElement , ieSix = (!window.XMLHttpRequest);
	var callback=callback||{okBack:function(){return true;},noBack:function(){return false;}};
	function gid(id){return d.getElementById(id);}
	if(!!gid("docinAlert")){		
		d.body.removeChild(gid("docinAlert"));
	}
	obj = d.createElement("div");	
	obj.className="docinPop";
	obj.id="docinAlert";	
	tempStr.push('<div><table class="docinPopTable"><tbody><tr><td class="pop_topleft"></td><td class="pop_border"></td><td class="pop_topright"></td></tr><tr><td class="pop_border"></td><td class="pop_content"><iframe id="alertIframe" scrolling="no" frameborder="0" style="height:100%;width:100%;position:absolute;top:0;left:0;z-index:-1;filter:(opacity=0);background:none;" ></iframe><div class="dialog_content" id="alertBody">');
	tempStr.push('<h3 class="dialog_title">豆丁提示:<a class="dialog_closed" title="关闭" href="javascript:void(0);" id="alertClose"><img alt="关闭" src="'+picture_image_path_v1+'/images_cn/newDocin/popClose.gif"></a></h3>');
	tempStr.push('<div class="dialog_body" style="font-size:14px;padding:5px 0;text-align:center;"><span style="display:inline-block;margin: 0 auto;text-align:left;">'+str+'</span></div>');	
	tempStr.push('<div class="dialog_buttons" ><input type="button" id="alertOk" class="docinCubeBt" value="确定" />');
	if(!!!pro){
	tempStr.push('</div>');
	}else{
	tempStr.push('<input type="button" id="alertNo" class="docinCubeBt docinCubebtGray" style="margin-left:40px;" value="取消" /></div>');
	}	
	tempStr.push('</div></td><td class="pop_border"></td></tr><tr><td class="pop_bottomleft"></td><td class="pop_border"></td><td class="pop_bottomright"></td></tr></tbody></table></div>');
	obj.innerHTML=tempStr.join("");
	d.body.appendChild(obj);
	dialogBoxShadow();
	var os = obj.style;
	os.display="block";
	//var temptop = dEle.scrollTop>0?dEle.scrollTop+d.body.scrollTop:d.body.scrollTop+d.documentElement.scrollTop;
	var temptop = d.body.scrollTop+d.documentElement.scrollTop;
	os.left=(dEle.clientWidth-obj.clientWidth)/2+dEle.scrollLeft+"px"; 	
	//os.top=(dEle.clientHeight-obj.clientHeight)/2+"px";
	os.top=(dEle.clientHeight-obj.clientHeight)/2+dEle.scrollTop+d.body.scrollTop+"px";	
	if(ieSix){os.top=(dEle.clientHeight-obj.clientHeight)/2+temptop+"px";}
	os.position ="absolute";
	os.zIndex="100000";
	function fixed(){
		os.top=(dEle.clientHeight-obj.clientHeight)/2+dEle.scrollTop+d.body.scrollTop+"px";			
	}		
	if(ieSix){
		gid("alertIframe").style.height=gid("alertBody").offsetHeight+"px";
		gid("alertIframe").style.width=gid("alertBody").offsetWidth+"px";			
		addEV(window,"scroll",fixed);
	}else{
		addEV(window,"scroll",fixed);
	}
	function hideObj(){
		d.body.removeChild(obj);
		dialogBoxHidden();
		os.display="none";
		if(ieSix){
			window.detachEvent("onscroll",fixed);
		}
	}		
	gid("alertClose").onclick=function(){
		hideObj();
		if(!!callback.noBack){
			callback.noBack();
		}
		return false;
	};
	gid("alertOk").onclick=function(){
		hideObj();
		if(!!callback.okBack){
			callback.okBack();
		}
		return true;
	};
	if(!!pro){
		gid("alertNo").onclick=function(){
			hideObj();
		if(!!callback.noBack){
			callback.noBack();
		}
			return false;
		}
	}
	return true;
}

function queryBookHouseByUserId(userId, isSearch){
	var $keyword = jQuery("#myKeyword");
	var keyword = jQuery.trim($keyword.val());
	if(keyword.length == 0 || keyword == "查找该用户收藏/上传的文档"){
		alert("请输入关键字");
		$keyword.focus();
		return;
	}
	if(!isSearch){
		var href = window.location.href;
		jQuery.post("/app/user/ajaxsetbackurl", {
			'random': Math.random(),
			'backurl': href
		}, function(e){
			setTimeout(function(){
				document.forms['searchUserProductForm'].submit();
			}, 200);
		});
	}else{
		document.forms['searchUserProductForm'].submit();
	}
}
var timeId = null;
function clearKeyword(id){
	clearTimeout(timeId);
	if(id == null)
		return;
	var obj = document.getElementById(id);
	if(obj == null)
		return;
	var $keyword = jQuery(obj);
	$keyword.val("").focus();
	jQuery('#spanBackUrl').show();
}
function searchWordFocus(obj,defaultValue){
	clearTimeout(timeId);
	if(obj.value==defaultValue)
		obj.value='';
	obj.style.color='#333';
	jQuery('#spanBackUrl').show()
}
function searchWordBlur(obj,defaultValue){
	if(obj.value==''){
		clearTimeout(timeId);
		timeId = setTimeout(function(){
			obj.value=defaultValue;
			obj.style.color='#C7C7C7';
			jQuery('#spanBackUrl').hide();
		}, 200);
	}
}
function showLoginAndClick(id){
	if(typeof(id) == 'undefined' || id == null)
		return;
	var cook = new CookieClass();
	cook.expires = 10;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("remindClickId", id);
	
	//下载和收藏的cookie变为-1
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("booksaveClickId", -1);
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("downloadClickId", -1);
	
	cook.expires = 1;
	cook.domain = ".docin.com";
	cook.path = "/";
	cook.setCookie("payReadClickId", -1);
	
	showlogin();
}
function initClick(){
	if(typeof(jQuery) != "function")
		return;
	jQuery(function(){
		var cook = new CookieClass();
		var remindClickId = cook.getCookie("remindClickId");
		if(remindClickId != null && remindClickId != "" && remindClickId != "-1"){
			if(remindClickId.indexOf("mobileFavMes")>=0){
				jQuery("#mbPrivCon").hide();
				jQuery("#" + remindClickId).show();
			    jQuery("#" + remindClickId).trigger("click");
			}else if(remindClickId.indexOf("subscribeCartoon")>=0){
				var $obj = null;
				try {
					$obj = jQuery("#" + remindClickId);
				} catch (e) {
				}
				if($obj != null && $obj.attr("title") == "订阅到首页"){
					var obj = $obj.get(0);
					if(obj.click){
						obj.click();
					}
				}
			}else{
				var $obj = null;
				try {
					$obj = jQuery("#" + remindClickId);
				} catch (e) {
				}
				if($obj != null && ($obj.text() == "收藏到书房" || $obj.text() == "+添加到首页精选")){
					var click = 0;//用来记录是否已经处理点击
					var top = $obj.offset().top;
					top = top - 100;
					if(top < 0)
						top = 0;
					jQuery("html,body").animate({"scrollTop": (top + "px")}, "fast", function(e){
						var obj = $obj.get(0);
						if(obj.click && click == 0){
							obj.click();
							click = 1;
						}else if(click == 0){
							var evt = document.createEvent("MouseEvents");
							evt.initEvent("click", true, true);
							obj.dispatchEvent(evt);
							click = 1;
						}
					});
				}
			}
		}
		cook.expires = 1;
		cook.domain = ".docin.com";
		cook.path = "/";
		cook.setCookie("remindClickId", -1);
	})
}
initClick();


	/*返回顶部*/
(function(window){
	var link_appdown =new RegExp('mobile_web/index.jsp'),//移动下载页
	link_bookstore = new RegExp('shufang.docin.com/'),//书房
	link_topic = new RegExp('docin.com/t-'),//单个专题
	link_topiclist = new RegExp('topicshow.do'),//专题列表
	link_doudan =new RegExp( 'docin.com/d-'),//单个豆单页
	link_search = new RegExp('docin.com/search.do'),//搜索结果页
	link_list = new RegExp('docin.com/l-'),//分类列表页
	link_listindex = new RegExp('docin.com/list.html'),//分类首页
	link_docend = new RegExp('docin.com/p-'),//终极页
	link_doudanlist = new RegExp('docin.com/dl-'),//豆单列表页
	link_cartoon = new RegExp('docin.com/cartoon'),//漫画
	link_daily = new RegExp('docin.com/daily/index.do');//daily列表页
	var link_sendMessage = new RegExp('app/my/msg/sendToSys');//给豆丁发消息
	var doc_url = window.location.href;
	var dimcode_txt = '<a style="cursor:default;" href="javascript:void(0);" class="t">扫二维码</a><div class="downloadApp"><a href="javascript:void(0);" style="cursor:default;"><img title="获取二维码" alt="获取二维码"></a>用手机继续访问</div>';
	if(link_appdown.test(doc_url)||link_bookstore.test(doc_url)){
		dimcode_txt = '<a class="t" href="http://www.docin.com/mobile_web/index.jsp">扫二维码</a><div class="downloadApp"><a target="_blank" href="http://www.docin.com/mobile_web/index.jsp"><img title="获取二维码" alt="获取二维码"></a>安装豆丁书房APP</div>';
		 doc_url = 'http://a.app.qq.com/o/simple.jsp?pkgname=com.docin.zlibrary.ui.android';
	}
	else if(link_docend.test(doc_url)||link_listindex.test(doc_url)||link_list.test(doc_url)||link_search.test(doc_url)||link_topic.test(doc_url)||link_doudanlist.test(doc_url)||link_doudan.test(doc_url)||link_topiclist.test(doc_url)||link_cartoon.test(doc_url)){
		doc_url = window.location.href;
	}
	else if(link_daily.test(doc_url)){
		doc_url = 'http://www.docin.com/daily/index.do';
	}
	else{
		doc_url = 'http://www.docin.com';
	}
	if(link_sendMessage.test(window.location.href)){
		var oSideBar = jQuery('<div class="backToTop"></div>').appendTo(document.body).html('<div class="gobackTop"><a href="javascript:void(0);" class="t">返回顶部</a></div><div class="docinDownApp">'+dimcode_txt+'</div>');
	}
	else{
		var oSideBar = jQuery('<div class="backToTop"></div>').appendTo(document.body).html('<div class="gobackTop"><a href="javascript:void(0);" class="t">返回顶部</a></div><div class="advice"><a href="http://www.docin.com/newMessage/talksWithSys.do" target="_blank" class="t">意见反馈</a></div><div class="docinDownApp">'+dimcode_txt+'</div>');
	}
	jQuery(".docinDownApp").mouseover(function(){
		jQuery(".downloadApp").show();
		jQuery(".downloadApp a img").attr("src","/servlet/get2wm?from=1&url=" + encodeURIComponent(doc_url));
	});
	jQuery(".docinDownApp").mouseout(function(){
		jQuery(".downloadApp").hide();
	});
	var oSideBarW = oSideBar.outerWidth(),
	oJustGoTop = jQuery(".gobackTop");
	function setBarPosition(){
		var st = jQuery(document).scrollTop(), winh = jQuery(window).height(),winw = jQuery(window).width();
		var oRight = (winw-1130)/2-oSideBarW-20;
		oRight<120?jQuery(".downloadApp").css({"left":-111}):jQuery(".downloadApp").css({"left":39});
		if(oRight<0){oRight=0};
		(st > 180)? oJustGoTop.show(): oJustGoTop.hide(); 
		oSideBar.css({'right':oRight});
		//IE6
		if (!window.XMLHttpRequest) {
			var oRight = (winw-1130)/2-oSideBarW-77;
			if(oRight<0){oRight=0};
			oSideBar.css({'right':oRight});   
		}
		oSideBar.show();
	}
	window.setBarPosition = setBarPosition;
	jQuery(window).bind("scroll resize", setBarPosition);
	jQuery(".gobackTop").bind("click",function(){
		jQuery("html, body").animate({ scrollTop: 0 }, 120);
	});
	jQuery(function() {setBarPosition(); });
})(window);
/*仿微博小黄条通知*/
	function feedNoticeFix(param){
		var d = document;		
		var obj = d.getElementById("feedNotice");
		if(!!!obj){
			return;
		}	
		//obj.style.display="block";
		obj.style.top=param.top+"px";
		var InitAbsTop =new getXYWH(obj).showT;
		function changePos(){				
				scrolling=true;
				var scrollH = d.documentElement.scrollTop + d.body.scrollTop;
				var AbsTop = obj.getBoundingClientRect().top;
				var curAbsTop = new getXYWH(obj).showT;				
				if(scrollH>0){
					if(scrollH>InitAbsTop){
						obj.style.top=(param.top+scrollH-InitAbsTop)+"px";
					}else{
						obj.style.top=(param.top)+"px";
					}				
				}else{
						obj.style.top=(param.top)+"px";
				}		
		}
		// addEV(window,"scroll",changePos);		
		// addEV(window,"change",changePos);
	}	
	
//baidu share ###############
function renrenBdShare(js_pid,js_cid){
	jQuery("#a5").load("/app/rrzt/join");
	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&cid="+js_cid+"&conntype=renren&rand="+new Date().getTime());
	
}	
function qqZoneBdShare(js_pid,js_cid){
	jQuery("#a5").load("/app/qqzt/join");
	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&cid="+js_cid+"&conntype=qqzone&rand="+new Date().getTime());
	
}
function sinaWbBdShare(js_pid,js_cid){
 	jQuery("#a5").load("/app/wbzt/join");
 	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&cid="+js_cid+"&conntype=sina&rand="+new Date().getTime());
 	
}
function qqWbBdShare(js_pid,js_cid){
	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&cid="+js_cid+"&conntype=qqweibo&rand="+new Date().getTime());
		
}
function qqFdBdShare(js_pid,js_cid){
	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&cid="+js_cid+"&conntype=feixin&rand="+new Date().getTime());
}

function kaixinBdShare(js_pid,js_cid){
	jQuery.post("/jsp_cn/renrenShareCallback.jsp?product_id="+js_pid+"&rand="+new Date().getTime());
}
//baidu share end ###############
//输入框placeholder
jQuery.fn.placeholder = function(){
	var i = document.createElement('input'),
		placeholdersupport = 'placeholder' in i;
	if(!placeholdersupport){
		var inputs = jQuery(this);
		inputs.each(function(){
			var input = jQuery(this),
				text = input.attr('placeholder'),
				pdl = 0,
				height = input.outerHeight(),
				width = input.outerWidth(),
				placeholder = jQuery('<span class="phTips">'+text+'</span>');
			try{
				pdl = input.css('padding-left').match(/\d*/i)[0] * 1;
			}catch(e){
				pdl = 5;
			}
			placeholder.css({'margin-left':'5px','top': 0,'left': 0,'height':height,'line-height':height+"px",'position':'absolute','color': "#cecfc9", 'font-size' : "14px"});
			placeholder.click(function(){
				input.focus();
			});
			if(input.val() != ""){
				placeholder.css({display:'none'});
			}else{
				placeholder.css({display:'inline'});
			}
			placeholder.insertAfter(input);
			input.bind("keyup focus blur",function(){
				if(jQuery(this).val() != ""){
					placeholder.css({display:'none'});
				}else{
					placeholder.css({display:'inline'});
				}
			});
			// input.keyup(function(e){
			// 	if(jQuery(this).val() != ""){
			// 		placeholder.css({display:'none'});
			// 	}else{
			// 		placeholder.css({display:'inline'});
			// 	}
			// });
		});
	}
	return this;
}
//搜索类型切换
	function initDocinSearch(){
		var oSearchTab = jQuery("#searchTab"),
		aTriggerList = jQuery("#searchTab .search_tab");
		if(oSearchTab.length>0){
			oSearchTab.hover(function(){
				jQuery(this).addClass("hover");
			},function(){
				jQuery(this).removeClass("hover");
			});
			aTriggerList.bind("click",function(){
				aTriggerList.removeClass("selected");
				var curSearchType = jQuery(this).data("searchtype");
				if(curSearchType == "user"){
					jQuery(this).addClass("selected").insertBefore(aTriggerList.eq(0));
					jQuery("#searchType_banner").val("u");
				}
				else if(curSearchType == "doc"){
					jQuery(this).addClass("selected").insertBefore(aTriggerList.eq(1));
					jQuery("#searchType_banner").val("p");
				}
				oSearchTab.removeClass("hover");
				return false;
			});
		}
	}

//创建动态弹层
function CreateDocinDialog(){
	/*
*@id 该弹层id
*@title 可选，提示标题
*@content 正文内容(文字活函数返回值)
*@cancel 取消按钮 txt:按钮文字，value：true为显示，false为不显示,
*@confirm 确认按钮 txt:按钮文字，value：true为显示，false为不显示
*@init 进行内容初始化，绑定事件类
*@callBack 回调函数｛okBack:function(){},noBack:function(){}｝
*返回值，确定为true，取消和关闭都为false
*/
	this.config = {
		id:"",
		cls:"",
		title:"豆丁提示",
		content:"",
		cancel:{txt:"取消",value:true},
		confirm:{txt:"确定",value:true},
		button:1,//弹层是否有按钮 2：无
		shadow:1,//是否有遮罩  2：无
		init:function(){},
		callBack:{okBack:function(){return true;},noBack:function(){return false;}},
		closeCallBack:function(){}
	};
	if(arguments.length == 1){
		var parame = arguments[0];
		for(var i in parame){
			this.config[i] = parame[i]||this.config[i];
		}
	}
	this.createBody();
}
CreateDocinDialog.prototype={
	createBody:function(){
		var docinAlert = jQuery('<div id="'+this.config.id+'" class="docinDialogBox"></div>'),
		iframe = jQuery('<iframe scrolling="no" frameborder="0" style="height:100%;_height:150px;width:100%;position:absolute;top:0;left:0;z-index:-1;"></iframe>'),
		dialogBody = jQuery('<div class="dialogBody"></div>'),
		dialogTitle = jQuery('<div class="dialogTitle"><h3>'+this.config.title+'</h3><a href="javascript:void(0);" title="关闭" class="dialog_closed"></a></div>'),
		dialogCont = jQuery('<div class="dialogCont">'+this.config.content+'</div>');
		iframe.appendTo(docinAlert);
		dialogTitle.appendTo(dialogBody);
		dialogCont.appendTo(dialogBody);
		dialogBody.appendTo(docinAlert);
		docinAlert.addClass(this.config.cls);
		if(this.config.button == 1){
			var dialogBtn = jQuery('<div class="dialog_buttons"></div>');
			if(this.config.confirm.value){
			jQuery('<a id="j_confirm" href="javascript:;" class="btn_blue">'+this.config.confirm.txt+'</a>').appendTo(dialogBtn);
			}
			if(this.config.cancel.value){
				jQuery('<a id="j_cancel" href="javascript:;" class="btn_blue btn_gray">'+this.config.cancel.txt+'</a>').appendTo(dialogBtn);
			}
			dialogBtn.appendTo(dialogBody);
		}
		if(this.config.shadow == 1){
			this.setShadow();//添加遮罩
		}
		docinAlert.appendTo(document.body);
		this.bind();
		this.setCenter();
		this.config.init();
	},
	bind:function(){
		var _this = this;
		var oCloseCurDialog = jQuery("#"+this.config.id+" .dialog_closed");
		if(oCloseCurDialog.length>0){
			oCloseCurDialog.bind("click",function(){
				_this.closeDialog();
				if(!!_this.config.callBack.noBack){
					_this.config.callBack.noBack();
				}
				_this.config.closeCallBack();
				return false;
			});
		}
		if(jQuery("#j_confirm").length>0){
			jQuery("#j_confirm").on("click",function(){
				_this.closeDialog();
				if(!!_this.config.callBack.okBack){
					_this.config.callBack.okBack();
				}
				return true;
				
			});
		}
		if(jQuery("#j_cancel").length>0){
			jQuery("#j_cancel").on("click",function(){
				_this.closeDialog();
				if(!!_this.config.callBack.noBack){
					_this.config.callBack.noBack();
				}
				return false;
			});
		}
	},
	closeDialog:function(){
		var _this = this;
		jQuery("#"+_this.config.id).hide().remove();
		if(_this.config.shadow == 1){
			jQuery("#dialogShadow").remove();
		}
		_this.config.closeCallBack();
	},
	setCenter:function(){
		// var iWidth = jQuery(window).width(),
		// iHeight = jQuery(window).height(),
		// obj = jQuery("#"+this.config.id),
		// objWidth = obj.outerWidth(),
		// objHeight = obj.outerHeight(),
		// objTop = parseFloat((iHeight-objHeight)/2)+'px',
		// objLeft = parseFloat((iWidth-objWidth)/2)+'px';
		// obj.css({position:"fixed",top:objTop,left:objLeft});
		setObjCenter(this.config.id);
	},
	setShadow:function(){
		var iWidth = jQuery(window).width(),
		bodyHeight = jQuery(document.body).height()+"px";
		if(!!window.XMLHttpRequest){
			jQuery('<div id="dialogShadow" style="position:fixed;left:0;top:0;bottom:0;right:0;width:100%;background-color:#000;opacity:0.4;filter:alpha(opacity=40);z-index:101;"></div>').appendTo(document.body);
		}
		else{
			jQuery('<div id="dialogShadow" style="position:absolute;left:0;top:0;width:100%;height:'+bodyHeight+';opacity:0.4;filter:alpha(opacity=40);background-color:#000;z-index:101;"></div>').appendTo(document.body);
		}
		
	}
}

jQuery(function(){
	jQuery('input[placeholder]').placeholder();
	initDocinSearch();//初始化搜索
	if(jQuery("#topsearch").length>0){
		LoadJS("http://www.baidu.com/js/opensug.js",1);
	}
	endSearchTest();
});

function endSearchTest(){
	var oSearchBtn = jQuery("#searchForm .ico_btn_search");
	if(oSearchBtn.length == 0){return false;}
    jQuery('#topsearch').bind('input propertychange', function() {
    	if(jQuery(this).val()==""){
    		oSearchBtn.removeClass('ico_btn_active');
    	}
    	else{
    		oSearchBtn.addClass('ico_btn_active');
    	}
	});
	if(jQuery('#topsearch').val()!==""){
		oSearchBtn.addClass('ico_btn_active');
	}
}
//关闭弹层 有参数f时为需要关闭遮罩
function commonHideBox(id,f){
	if(f == 1){
		dialogBoxHidden();
		jQuery("#"+id).remove();
	}
	else{
		jQuery("#"+id).remove();
		if(id == "vipPermiss"){
			setCookie2008_1("vip_red_tip",100,360);
		}
	}
}

function commonConutHideBox(id,f,s){//f:是否有弹层 值：1   s:是否倒计时关闭 值：2
	if(jQuery('#'+id).length == 0){return false;}
	clearTimeout(setTimer);
	var args = arguments.length;
	if(args == 1){
		jQuery("#"+id).remove();
	}
	else if(args == 2){
		if(arguments[1] == 1){
			dialogBoxHidden();
			jQuery("#"+id).remove();
		}
		else if(arguments[1] == 2){
			setTimer = setTimeout(function(){
				jQuery("#"+id).fadeOut();
				jQuery("#"+id).remove();
			},3000);
		}
	}
	else if(args == 3){
		setTimer = setTimeout(function(){
			jQuery("#dialogShadow").remove();
			jQuery("#"+id).fadeOut();
			jQuery("#"+id).remove();
		},3000);
	}
}
//vip调查
var investConfig = {
	'id':'docinInvest',
	'cls':'invertForm',
	'title':'感谢使用豆丁网[用户调查]',
	'button':2,
	'content':(function(){var tempDiv = jQuery('<div></div>');
		jQuery('<p class="titInfo"><span id="j_vip_username"></span>,  我们挑选了部分重要的用户做此调查, 你的选择对我们很重要, 也会帮助我们更精确地为你呈现内容。</p>').appendTo(tempDiv);
		var investCon = jQuery('<div class="Con"></div>').appendTo(tempDiv);
		jQuery('<div class="l"> 请勾选以下你关注的方向:</div>').appendTo(investCon);
		jQuery('<ul><li><label><input type="checkbox" value="1" name="t">企业管理，创业过程相关资料</label></li><li><label><input type="checkbox" value="2" name="t">行业数据，各类贸易数据</label></li><li><label><input type="checkbox" value="3" name="t">培训课程，学校教育</label></li><li><label><input type="checkbox" value="4" name="t">各类资格考证，认证，公务员备考</label></li><li><label><input type="checkbox" value="5" name="t">法律法规，规章制度等规范性文件</label></li><li><label><input type="checkbox" value="6" name="t">范文模版，合同样本，示例表格</label></li><li><label><input type="checkbox" value="7" name="t">新闻评论，社会资讯</label></li><li><label><input type="checkbox" value="8" name="t">保健知识，健康养生</label></li><li><label><input type="checkbox" value="9" name="t">孕妇课堂，育儿知识，亲子教育</label></li><li>其他<label class="other"><textarea id="otherRe"></textarea></label></li></ul>').appendTo(investCon);
			jQuery('<div class="dialog_buttons"><a class="btn_blue" href="javascript:;" id="j_confirm">提交调查</a><a class="btn_blue btn_gray" href="javascript:;" id="j_cancel">残忍拒绝</a></div><div id="investTips" style="color:#EE0000;margin:5px 0 0 0;display:none;">提交失败，请至少填选一项</div>').appendTo(investCon);
			return tempDiv.html();})(),
	callBack:{
			okBack:function(){
				function btnBling(){
					var t = 3;
					var shockTips = setInterval(function(){
						if(t==0){
							clearInterval(shockTips);
						}
						if(t%2 == 1){jQuery("#j_confirm").css('backgroundColor',"#ff0000");}
						else{
							jQuery("#j_confirm").css('backgroundColor',"#3086F2");
						}
						t--;
					},200);
				}
				//点击确认或完成时回调函数
				var se = jQuery("input[name='t']:checked"),prama = [],otherRe =jQuery("#otherRe").val();
				if(se.length<1&&otherRe==""){
					btnBling();
					jQuery("#investTips").html('提交失败：请至少填选一项').show();
					return false;
				}else if(otherRe.length>500){
					btnBling();
					jQuery("#investTips").html('提交失败：其他项限制500字符以内').show();
					return false;
				}
				se.each(function(i){
					prama[i] = jQuery(this).val();

				});
				jQuery.post("/app/dcwj/ajax/questionnaire",{checkFlag:prama.join('_'),other:otherRe},function(data){
					if(data == 2){
						jQuery("#docinInvest").remove();
						jQuery("#dialogShadow").remove();
					}
					else if(data == 3||data == 5){
						setCookie2008_1("alert_dcwj",user_id+"_1",180);
						docinResearch.closeDialog();
					}
					else if(data == 4){
						jQuery("#investTips").html('提交失败：请重新提交').show();
					}
				});

			},
			noBack:function(){
				//点击关闭或取消时回调函数
			}
		},
	closeCallBack:function(){
		setCookie2008_1("alert_dcwj",user_id+"_1",365);
	},
	init:function(){
		var oVipUserName = jQuery("#j_vip_username");
		if(oVipUserName.length>0){
			oVipUserName.html(user_name);
		}
	}
};
//json  {"noticeCount":公告数,"informCount":通知数,"privateCount":私信数,"totalCount":总提醒数,"remind":提示条}	
function getRealTimeNews(){
	var totalMessage = informNum = privateNum = noticeNum = -1;
	var realTimer = null;
	getMessageInfo();
	// realTimer = setInterval(function(){
	// 	getMessageInfo();
	// },50000);
	function setMessagePage(data){
		var aMessageBox = jQuery(".unread_mess");
		if(aMessageBox.length==0){return;}
		if(data.noticeCount > 0 && data.noticeCount != noticeNum){
			aMessageBox.eq(0).show().html(data.noticeCount);
		}
		else if(data.noticeCount == 0){
			aMessageBox.eq(0).hide();
		}
		if(data.informCount > 0 && data.informCount != informNum){
			aMessageBox.eq(1).show().html(data.informCount);
		}
		else if(data.informCount == 0){
			aMessageBox.eq(1).hide();
		}
		if(data.privateCount > 0 && data.privateCount != privateNum){
			aMessageBox.eq(2).show().html(data.privateCount);
		}
		else if(data.privateCount == 0){
			aMessageBox.eq(2).hide();
		}
	}
	function getMessageInfo(){
		var aBannerMess = jQuery(".red_unread");
		if(aBannerMess.length == 0){return false;}
		jQuery.ajax({
			url:'/newMessage/newMessageCountAndRemind.do',
			cache:false,
			success:function(data){
				var data = jQuery.parseJSON(data);
				setMessagePage(data);
				if(totalMessage != data.totalCount){
					totalMessage = data.totalCount;
					noticeNum = data.noticeCount;
					informNum = data.informCount;
					privateNum = data.privateCount;
					remind = data.remind;
				}
				else{
					return false;
				}
				if(data.totalCount>0){
					jQuery(".red_unread").show().html(data.totalCount);
					if(data.informCount>0){
						var messageCountUrl = "/newMessage/informMessage.do";//通知
					}
					else if(data.noticeCount>0){
						var messageCountUrl = "/newMessage/noticeMessage.do";//公告
						aBannerMess.parent().attr({href:messageCountUrl});
					}
					else if(data.privateCount>0){
						var messageCountUrl = "/newMessage/privateMessage.do";//私信
						aBannerMess.parent().attr({href:messageCountUrl});
					}

					if(remind.indexOf(',')>-1){
						ajaxGetMessageFlat();
					}
				}
				else{
					jQuery(".red_unread").hide().html();
					var messageCountUrl = "/app/user/editUserInfo";//头像编辑
					aBannerMess.eq(0).parent().attr({href:messageCountUrl});
				}

			}

		});
	}
}
function openSearch(searchText){
	if(typeof(spider_name) == "undefined"){return false;}
	var searchText = encodeURIComponent(searchText);
	var openUrl = 'http://www.docin.com/app/p/end/ajax/spiderJump.do?s='+spider_name+'&k='+searchText;
	return openUrl;
}
//请求广告代码
function docin_adload (type,parentId) {
		//type 0:img 1:text 3:code
		if(jQuery("#"+parentId).length == 0){return false;}
		var moneyUrl = '/docin_adv/adv.do?pos='+type;       
		var oIframe = jQuery('<iframe width="100%" height="100%" scrolling="no" frameborder="0" style="display:block;margin:0 auto;" src="'+moneyUrl+'"></iframe>').appendTo(jQuery("#"+parentId));
	}
jQuery(document).ready(function(){
			getRealTimeNews();//请求消息
			var channelTimer = null;
			if(jQuery(".channel_mod").length>0){
				jQuery(".channel_mod").bind("mouseover",function(){
					clearTimeout(channelTimer);
					jQuery(".channel_mod").addClass('channel_mod_open');
				});
				jQuery(".channel_mod").bind("mouseleave",function(){
					channelTimer = setTimeout(function(){
						jQuery(".channel_mod").removeClass('channel_mod_open');
					},300);
				});
			}
});
//邮箱未验证弹出层阻止
function mailDialogAlert(a,landingEmail,domailUrl){
		var mailDiv = jQuery('<div id="mailStopDiv" class="docinDialogBox mailStopDiv"><iframe frameborder="0" style="height:100%;_height:150px;width:100%;position:absolute;top:0;left:0;z-index:-1;" scrolling="no"></iframe><div id="dialogBody" class="dialogBody"></div></div>').appendTo(document.body);
		var mailCont = jQuery('<a class="closeX" onclick="commonHideBox(\'mailStopDiv\',1);" href="javascript:void(0);" title="关闭"></a><dl><dt><img src="'+domailUrl+'/images_cn/ico_warn_tips.png" alt="" />验证您的邮箱，方可使用该功能</dt><dd>最后一步……</dd><dd>系统已经发送了一封验证邮件到您的邮箱'+a+' <a href="'+domailUrl+'/app/my/docin/validate" title="修改邮箱" target="_blank">修改邮箱</a><br/>点击邮件中的链接即可完成验证。（验证成功将获得1豆元奖励）</dd><dd class="mart12"><a title="前往邮箱" href="'+landingEmail+'" class="ico_button btnc_blue" target="_blank">前往邮箱</a><br /></dd><dd id="sendMail_1" class="p3"><a class="send_again" href="javascript:createuser_sendMail_a();" onclick="sendMail();" title="点此重新发送验证邮件">点此重新发送验证邮件</a></dd><dd id="sendMail_2" class="p4" style="display:none;">(<span id="sendSec"></span>)重新发送</dd><dd id="sendMail_3" class="p5" style="display:none;"><p class="mail_help">一直没有收到验证邮件怎么办？</p><p>· 看看是否在垃圾邮箱或订阅邮箱里</p><p>· 建议您10分钟后重新尝试</p></dd></dl>').appendTo(jQuery("#dialogBody"));
		dialogBoxShadow();
		setObjCenter("mailStopDiv");
}
//邮箱发送倒计时
function sendMailTimer(id){
	var o = jQuery("#"+id),wait = 60,waitTimer = null;
	if(o.length == 0){return;}
	time();
	createuser_sendMail_a();
	//校验码倒计时
	function time() {
		if (wait == 0) {
			o.val("重新发送");
			wait = 60;
		} else {
			o.attr("disabled",true);
			o.val('('+wait+')'+'重新发送');
			wait--;
			waitTimer = setTimeout(function() {
				time();
			},
			1000);
		}
	}
}
function createuser_sendMail_a(){
	var url = "/dwrutil/sendMailAgain_new.do";
	jQuery.get(url,null,function(data){
		var falg = data == "true";
	});
}	
/*点击重新发送邮件*/
(function(window){
	var waitTimer = null,wait = 60;
	//校验码倒计时
	function time(id) {
		var o = jQuery("#"+id);
		if(o.length == 0){return;}
		if (wait == 0) {
			jQuery("#sendMail_1").show();
			jQuery("#sendMail_2").hide();
			wait = 60;
		} else {
			o.html(wait);
			wait--;
			waitTimer = setTimeout(function() {
				time(id);
			},
			1000);
		}
	}
	function sendMail(){
		jQuery("#sendMail_1").hide();
		time("sendSec");
		jQuery("#sendMail_2,#sendMail_3").show();
	}
	window.sendMail = sendMail;
})(window);