function Div(el)
  if el.classes:includes('proof_idea') or el.classes:includes('proofbox') then
    
    local box_type = 'proof'
    local title = 'Proof'
    
    if el.classes:includes('proof_idea') then
      box_type = 'idea'
      title = 'Proof Idea'
    end
    
    if el.attributes['title'] then
      title = el.attributes['title']
    end
    
    -- Recreate EXACTLY the raw HTML you just tested successfully
    local open_tag = string.format('<details class="beamer-box %s">\n<summary>%s</summary>\n\n', box_type, title)
    local close_tag = '\n\n</details>'
    
    -- Build a new list of blocks
    local result = pandoc.List()
    result:insert(pandoc.RawBlock('html', open_tag))
    
    -- Insert the actual math and text you wrote in the markdown
    result:extend(el.content)
    
    result:insert(pandoc.RawBlock('html', close_tag))
    
    -- Returning this list completely deletes the ::: and replaces it with our HTML
    return result
    
    elseif el.classes:includes('box') then
    if el.attributes['title'] then
      local title_html = string.format('<div class="box-title">%s</div>', el.attributes['title'])
      local result = pandoc.List()
      result:insert(pandoc.RawBlock('html', title_html))
      result:extend(el.content)
      el.content = result
    end
    return el
  end
end
