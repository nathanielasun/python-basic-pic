(function () {
  "use strict";

  function initPanels() {
    document.querySelectorAll(".panel").forEach(function (panel) {
      var header = panel.querySelector(".panel-header");
      if (!header) return;

      header.setAttribute("aria-expanded", panel.classList.contains("open") ? "true" : "false");

      header.addEventListener("click", function () {
        var isOpen = panel.classList.toggle("open");
        header.setAttribute("aria-expanded", isOpen ? "true" : "false");
      });
    });
  }

  function initSidebar() {
    var links = Array.from(document.querySelectorAll(".sidebar a[href^='#']"));
    var sections = links
      .map(function (link) {
        var id = link.getAttribute("href").slice(1);
        var section = document.getElementById(id);
        return section ? { link: link, section: section } : null;
      })
      .filter(Boolean);

    function setActive() {
      var scrollPos = window.scrollY + 120;
      var current = sections[0];

      sections.forEach(function (entry) {
        if (entry.section.offsetTop <= scrollPos) {
          current = entry;
        }
      });

      links.forEach(function (link) {
        link.classList.remove("active");
      });
      if (current) {
        current.link.classList.add("active");
      }
    }

    window.addEventListener("scroll", setActive, { passive: true });
    setActive();
  }

  function expandPanelsWithHash() {
    var hash = window.location.hash;
    if (!hash) return;
    var target = document.querySelector(hash);
    if (!target) return;
    var panel = target.closest(".panel");
    if (panel) {
      panel.classList.add("open");
      var header = panel.querySelector(".panel-header");
      if (header) header.setAttribute("aria-expanded", "true");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    initPanels();
    initSidebar();
    expandPanelsWithHash();
  });
})();
