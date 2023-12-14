/*******************************************************************************
 * Set a custom icon for pypi as it's not available in the fa built-in brands
 * Taken from: https://github.com/pydata/pydata-sphinx-theme/blob/main/docs/_static/custom-icon.js
 */
FontAwesome.library.add(
  (faListOldStyle = {
    prefix: "fa-custom",
    iconName: "pypi",
    icon: [
      17.313, // viewBox width
      19.807, // viewBox height
      [], // ligature
      "e001", // unicode codepoint - private use area
      "m10.383 0.2-3.239 1.1769 3.1883 1.1614 3.239-1.1798zm-3.4152 1.2411-3.2362 1.1769 3.1855 1.1614 3.2369-1.1769zm6.7177 0.00281-3.2947 1.2009v3.8254l3.2947-1.1988zm-3.4145 1.2439-3.2926 1.1981v3.8254l0.17548-0.064132 3.1171-1.1347zm-6.6564 0.018325v3.8247l3.244 1.1805v-3.8254zm10.191 0.20931v2.3137l3.1777-1.1558zm3.2947 1.2425-3.2947 1.1988v3.8254l3.2947-1.1988zm-8.7058 0.45739c0.00929-1.931e-4 0.018327-2.977e-4 0.027485 0 0.25633 0.00851 0.4263 0.20713 0.42638 0.49826 1.953e-4 0.38532-0.29327 0.80469-0.65542 0.93662-0.36226 0.13215-0.65608-0.073306-0.65613-0.4588-6.28e-5 -0.38556 0.2938-0.80504 0.65613-0.93662 0.068422-0.024919 0.13655-0.038114 0.20156-0.039466zm5.2913 0.78369-3.2947 1.1988v3.8247l3.2947-1.1981zm-10.132 1.239-3.2362 1.1769 3.1883 1.1614 3.2362-1.1769zm6.7177 0.00213-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4124 1.2439-3.2947 1.1988v3.8254l3.2947-1.1988zm-6.6585 0.016195v3.8275l3.244 1.1805v-3.8254zm16.9 0.21143-3.2947 1.1988v3.8247l3.2947-1.1981zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm-3.4124 1.2432-3.2947 1.1988v3.8254l3.2947-1.1988zm-6.6585 0.019027v3.8247l3.244 1.1805v-3.8254zm13.485 1.4497-3.2947 1.1988v3.8247l3.2947-1.1981zm-3.4145 1.2411-3.2926 1.2016v3.8247l3.2926-1.2009zm2.4018 0.38127c0.0093-1.83e-4 0.01833-3.16e-4 0.02749 0 0.25633 0.0085 0.4263 0.20713 0.42638 0.49826 1.97e-4 0.38532-0.29327 0.80469-0.65542 0.93662-0.36188 0.1316-0.65525-0.07375-0.65542-0.4588-1.95e-4 -0.38532 0.29328-0.80469 0.65542-0.93662 0.06842-0.02494 0.13655-0.03819 0.20156-0.03947zm-5.8142 0.86403-3.244 1.1805v1.4201l3.244 1.1805z", // svg path (https://simpleicons.org/icons/pypi.svg)
    ],
  })
);
