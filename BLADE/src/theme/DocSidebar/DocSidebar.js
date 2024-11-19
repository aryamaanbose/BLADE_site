import React from 'react';
import SidebarSearch from '@site/src/components/SidebarSearch';
import { useDocsSidebar } from '@docusaurus/theme-common';

function DocSidebar() {
  const sidebar = useDocsSidebar();

  // Transform the sidebar items to pass to SidebarSearch
  const sidebarItems = sidebar.items.map((item) => ({
    label: item.label,
    href: item.href,
  }));

  return (
    <div>
      <SidebarSearch items={sidebarItems} />
      {/* The rest of your sidebar content goes here */}
    </div>
  );
}

export default DocSidebar;
